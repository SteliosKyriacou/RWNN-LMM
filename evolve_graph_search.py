"""
Fully-free atomic graph search (Claude-emitted graph edits).

The genome is the raw (nodes, edges) H-DAG. Each generation the Metis agent (Claude)
is shown the current Pareto front and emits a GRAPH-EDIT PROGRAM built from *generic
atomic operations only* — add_node / add_edge / remove_node / remove_edge — over the
primitive vocabulary of rwnn/nodes.py. There are NO named "attention"/"FFN"/"MoE"
blocks: the agent composes arbitrary graphs from atoms, so a mixture-of-experts, a
gate, or an entirely novel mixer must EMERGE from wiring (e.g. linear -> softmax ->
element_mul -> sum). `causal_attention` is kept as one primitive token-mixing atom.

Only the I/O is fixed: token+positional embeddings in (nodes 0-3), LM head out (node 13).
Everything between is free. Every edited graph is pruned to the input->head live set and
DAG-validated; anything that still fails at runtime is rejected by the trainer.

Objectives (minimize): (1) validation loss  (2) active-FLOPs / token.
Hard constraints (reject-by-design): peak mem < 11 GB, loss < 4.5, >= 3 compute atoms.
"""
import os, json, re, copy
from collections import defaultdict, deque, Counter
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from rwnn.graph import RWNNGraph
from rwnn.mutator import is_valid_dag
from agentic_optimizer import claude_llm
from evolve_agentic import (train_and_eval_bpe_model, build_initial_population,
                            vector_to_multilayer_graph)

VOCAB, BLOCK, DMODEL = 50257, 256, 768
LOSS_MAX, MIN_COMPUTE, MEM_BUDGET, PENALTY = 4.5, 3, 11.0e9, 10.0
CKPT = "checkpoints/graph-search"
POP = 20
ANCHORS = {0, 1, 2, 3, 13}   # input, token_emb, pos_emb, emb_sum, head — never removed

# primitive vocabulary the agent may instantiate, with default kwargs filled in if omitted
PRIMS = {
    "linear":          {"d_in": DMODEL, "d_out": DMODEL},
    "layer_norm":      {"d_model": DMODEL},
    "activation":      {"act_type": "gelu"},
    "softmax":         {},
    "sum":             {},
    "element_mul":     {},
    "concat":          {"dim": -1},
    "causal_attention":{"n_head": 12, "d_model": DMODEL, "dropout": 0.1},
    "matmul":          {"d_in": DMODEL, "d_out": DMODEL},
    "scale_shift":     {"d_model": DMODEL},
    "add_bias":        {"d_model": DMODEL},
    "mean_reduce":     {"dim": -1},
    "dropout":         {"dropout": 0.1},
}


# ----------------------------------------------------------------------------- graph utilities
def _rid(nodes):
    return max((n["id"] for n in nodes), default=0) + 1

def _prune_to_live(nodes, edges):
    """Keep only nodes on an input(0) -> head(13) path."""
    adj, radj = defaultdict(list), defaultdict(list)
    for u, v in edges:
        adj[u].append(v); radj[v].append(u)
    def bfs(start, g):
        seen = {start}; q = deque([start])
        while q:
            x = q.popleft()
            for y in g[x]:
                if y not in seen:
                    seen.add(y); q.append(y)
        return seen
    ids = {n["id"] for n in nodes}
    if 0 not in ids or 13 not in ids:
        return None, None
    live = bfs(0, adj) & bfs(13, radj)
    if 13 not in live:
        return None, None
    return ([n for n in nodes if n["id"] in live],
            [(u, v) for u, v in edges if u in live and v in live])

def apply_edits(parent, ops):
    """Apply a list of atomic edits to a copy of parent; revert entirely if the result is invalid."""
    g = copy.deepcopy(parent)
    nodes, edges = g["nodes"], g["edges"]
    label2id = {}
    def resolve(r):
        return label2id.get(r) if isinstance(r, str) else r
    for op in ops if isinstance(ops, list) else []:
        try:
            k = op.get("op")
            if k == "add_node":
                t = op.get("type")
                if t not in PRIMS:
                    continue
                nid = _rid(nodes)
                kw = dict(PRIMS[t]); kw.update(op.get("kwargs") or {})
                nodes.append({"id": nid, "type": t, "kwargs": kw})
                if isinstance(op.get("id"), str):
                    label2id[op["id"]] = nid
                for r in (op.get("inputs") or []):
                    u = resolve(r)
                    if u is not None:
                        edges.append((u, nid))
            elif k == "add_edge":
                u, v = resolve(op.get("u")), resolve(op.get("v"))
                if u is not None and v is not None and u != v and (u, v) not in edges:
                    edges.append((u, v))
            elif k == "remove_edge":
                u, v = resolve(op.get("u")), resolve(op.get("v"))
                edges[:] = [e for e in edges if e != (u, v)]
            elif k == "remove_node":
                rn = resolve(op.get("id"))
                if rn in ANCHORS or rn is None:
                    continue
                nodes[:] = [n for n in nodes if n["id"] != rn]
                edges[:] = [(u, v) for u, v in edges if u != rn and v != rn]
        except Exception:
            continue
    ids = {n["id"] for n in nodes}
    edges[:] = [(u, v) for u, v in edges if u in ids and v in ids]
    pn, pe = _prune_to_live(nodes, edges)
    if pn is None or not is_valid_dag(pn, pe) or 13 not in {n["id"] for n in pn}:
        return copy.deepcopy(parent)
    if len([n for n in pn if n["type"] in ("causal_attention", "linear", "moe_ffn", "matmul")]) < MIN_COMPUTE + 1:
        return copy.deepcopy(parent)
    return {"nodes": pn, "edges": pe}


def active_flops(nodes):
    T, f = BLOCK, 0.0
    for n in nodes:
        t, kw = n["type"], n.get("kwargs", {})
        if t == "causal_attention":
            d = kw.get("d_model", DMODEL); f += 2 * (4 * d * d) + 2 * (2 * T * d)
        elif t == "linear":
            if n["id"] == 13:  # exclude constant LM head
                continue
            f += 2 * kw.get("d_in", DMODEL) * kw.get("d_out", DMODEL)
        elif t == "matmul":
            f += 2 * kw.get("d_in", DMODEL) * kw.get("d_out", DMODEL)
        elif t == "moe_ffn":
            d, h = kw["d_model"], kw["d_hidden"]; f += 2 * (d * kw["n_experts"]) + kw["top_k"] * 2 * (2 * d * h)
        elif t == "causal_batch_matmul":
            f += 2 * T * DMODEL
    return f

def summarize(g):
    nodes, edges = g["nodes"], g["edges"]
    hist = Counter(n["type"] for n in nodes)
    sm = {n["id"] for n in nodes if n["type"] == "softmax"}
    em = {n["id"] for n in nodes if n["type"] == "element_mul"}
    gating = sum(1 for u, v in edges if u in sm and v in em)         # emergent gate pattern
    sum_ids = [n["id"] for n in nodes if n["type"] == "sum" and n["id"] != 3]
    indeg = Counter(v for _, v in edges)
    skips = sum(max(0, indeg.get(s, 0) - 2) for s in sum_ids)
    compute = hist.get("causal_attention", 0) + sum(1 for n in nodes if n["type"] == "linear" and n["id"] != 13)
    return dict(attn=hist.get("causal_attention", 0),
                linear=sum(1 for n in nodes if n["type"] == "linear" and n["id"] != 13),
                softmax=hist.get("softmax", 0), elemmul=hist.get("element_mul", 0),
                gating=gating, skips=skips, compute=compute,
                nodes=len(nodes), edges=len(edges), hist=dict(hist))

def params_of(g):
    m = RWNNGraph(g["nodes"], g["edges"], global_d_model=DMODEL)
    p = sum(x.numel() for x in m.parameters()); del m
    return p

def graph_sig(g):
    return json.dumps([sorted((n["id"], n["type"]) for n in g["nodes"]),
                       sorted(tuple(e) for e in g["edges"])], sort_keys=True)

def anchors_of(g):
    """Return (tail feeding head-LN, ln_f id, list of sum-node ids) for the agent to attach edits to."""
    ln_f = next((u for u, v in g["edges"] if v == 13), None)
    tail = next((u for u, v in g["edges"] if v == ln_f), None)
    sums = [n["id"] for n in g["nodes"] if n["type"] == "sum" and n["id"] != 3]
    return tail, ln_f, sums


def seed_gates_and_skips(g, rng):
    """Seed a graph (in the initial population) with an emergent softmax gate + a SwiGLU-style gate
    and a couple of random long-range skip connections. Built from existing atoms only."""
    tail, ln_f, _ = anchors_of(g)
    if tail is None or ln_f is None:
        return g
    act = str(rng.choice(["gelu", "silu", "relu"]))
    ops = [
        {"op": "add_node", "id": "v",  "type": "linear",     "kwargs": {"d_out": DMODEL}, "inputs": [tail]},
        {"op": "add_node", "id": "gl", "type": "linear",     "kwargs": {"d_out": DMODEL}, "inputs": [tail]},
        {"op": "add_node", "id": "gg", "type": "activation", "kwargs": {"act_type": act}, "inputs": ["gl"]},
        {"op": "add_node", "id": "sw", "type": "element_mul", "inputs": ["v", "gg"]},        # SwiGLU-style gate
        {"op": "add_node", "id": "sl", "type": "linear",     "kwargs": {"d_out": DMODEL}, "inputs": [tail]},
        {"op": "add_node", "id": "sm", "type": "softmax",    "inputs": ["sl"]},
        {"op": "add_node", "id": "sg", "type": "element_mul", "inputs": ["v", "sm"]},        # softmax gate
        {"op": "add_node", "id": "gm", "type": "sum",        "inputs": [tail, "sw", "sg"]},
        {"op": "remove_edge", "u": tail, "v": ln_f},
        {"op": "add_edge", "u": "gm", "v": ln_f},
    ]
    g = apply_edits(g, ops)
    sums = anchors_of(g)[2]                                   # add 1-2 random long-range skips
    if len(sums) >= 3:
        for _ in range(int(rng.randint(1, 3))):
            i = int(rng.randint(0, len(sums) - 2)); j = int(rng.randint(i + 1, len(sums)))
            g = apply_edits(g, [{"op": "add_edge", "u": sums[i], "v": sums[j]}])
    return g


# ----------------------------------------------------------------------------- Claude atomic-edit agent
SYSTEM = """You are Metis-Graph, an autonomous neural-architecture search agent. You evolve a
decoder-only language model whose architecture is a directed ACYCLIC graph of PRIMITIVE atoms.
You do not write the network directly — you emit a GRAPH-EDIT PROGRAM that transforms parent
graphs on the current Pareto front, using ONLY these generic atomic operations:

  {"op":"add_node","id":"<label>","type":"<primitive>","kwargs":{...},"inputs":[<node refs>]}
  {"op":"add_edge","u":<ref>,"v":<ref>}
  {"op":"remove_edge","u":<ref>,"v":<ref>}
  {"op":"remove_node","id":<real node id>}

A <ref> is either a real integer node id (shown to you in the context) or a "<label>" you assigned
to a node created earlier in the SAME program. `inputs` wires predecessors -> the new node.

## Primitive vocabulary (these are the ONLY node types; there are NO attention/FFN/MoE blocks)
  linear {d_in,d_out}   -- affine map; the compiler AUTO-PROJECTS mismatched dims, so wiring is forgiving
  layer_norm {d_model}
  activation {act_type: gelu|silu|relu}
  softmax {}            -- over the last dim (use to build gates/routers)
  element_mul {}        -- elementwise product of its inputs (a width-1 input broadcasts) -> gating
  sum {}                -- elementwise add of ALL inputs -> residual/merge; extra inputs = skip connections
  concat {dim:-1}
  causal_attention {n_head:12,d_model:768}   -- one masked multi-head self-attention atom
  matmul {d_in,d_out}, scale_shift {d_model}, add_bias {d_model}, mean_reduce {dim}, dropout {dropout}

## Fixed anchors (never remove): 0 input, 1 token_emb, 2 pos_emb, 3 emb_sum, 13 LM head.
Everything between node 3 and node 13 is yours to (re)build. For each elite you are given its
current head-feeding node (`tail`), its final layer-norm id (`ln_f`), and its residual sum-node ids.

## Emergence, not templates
Nothing is pre-built. If you want a feed-forward, wire linear->activation->linear and a `sum` for
its residual. If you want a mixture-of-experts, build a router (linear->softmax) and gate several
linear "experts" with element_mul, then sum them. If you want something nobody has tried, wire it.
Two objectives are MINIMIZED (Pareto): (1) validation loss, (2) active-FLOPs/token. Extra parallel
compute costs FLOPs; skips and gates are cheap. Reject-triggers: peak mem >12GB, loss>=4.5, <3 compute atoms.

## Output (STRICT): 3-4 short LEARNINGS bullets, then a fenced ```json block: a list of exactly
{pop_size} children, each {"parent": <elite index, or -1 for a random seed>, "ops":[ ...atomic ops... ]}.
An empty ops list re-trains a parent unchanged. Wire every new subgraph so it reaches node 13."""

def build_context(gen, front, hist):
    lines = [f"Generation {gen}. Pareto front ({len(front)} elites), sorted by active-FLOPs:"]
    for i, e in enumerate(sorted(front, key=lambda z: z["flops"])):
        s = e["summary"]; t, lnf, sums = e["anchors"]
        lines.append(f"  [elite {i}] loss {e['loss']:.3f}, {e['flops']/1e9:.3f}G, {e['params']/1e6:.0f}M | "
                     f"atoms: attn {s['attn']}, linear {s['linear']}, softmax {s['softmax']}, "
                     f"element_mul {s['elemmul']}, gates {s['gating']}, skips {s['skips']}")
        lines.append(f"           anchors: tail(head-input)={t}, ln_f={lnf}, sum_ids={sums[:24]}")
    if hist:
        lines.append("\nBest loss per generation: " + ", ".join(f"{h:.3f}" for h in hist[-8:]))
    lines.append(f"\nEmit a graph-edit program producing exactly {POP} children.")
    return "\n".join(lines)

def ask_edits(gen, front, hist, msgs):
    msgs.append({"role": "user", "text": build_context(gen, front, hist)})
    prompt = "\n\n".join(f"===== {'REQUEST' if m['role']=='user' else 'YOUR RESPONSE'} =====\n{m['text']}"
                         for m in msgs[-10:])
    text, thinking = claude_llm.invoke(SYSTEM.replace("{pop_size}", str(POP)), prompt,
                                       model=os.environ["CLAUDE_MODEL"])
    if thinking:
        print(f"\n  \033[32m[THINKING] {thinking.strip()[:1200]}\033[0m", flush=True)
    msgs.append({"role": "assistant", "text": text})
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    raw = m.group(1) if m else (text[text.find("["):text.rfind("]") + 1] if "[" in text else "")
    try:
        prog = json.loads(raw); assert isinstance(prog, list); return prog
    except Exception as e:
        print(f"  [WARN] unparseable edit program ({e}); reusing parents unchanged", flush=True)
        return None


# ----------------------------------------------------------------------------- main loop
def run_graph_search(generations=100, pop_size=20, eval_steps=57860):
    global POP; POP = pop_size
    import shutil
    if os.path.exists(CKPT): shutil.rmtree(CKPT)
    os.makedirs(CKPT, exist_ok=True)
    print("=== FULLY-FREE ATOMIC GRAPH SEARCH STARTED ===", flush=True)

    seeds = build_initial_population(pop_size)
    seed_graphs = []
    for x in seeds:
        n, e = vector_to_multilayer_graph(x, VOCAB, BLOCK, d_model=DMODEL)
        seed_graphs.append({"nodes": [dict(nd) for nd in n], "edges": [tuple(t) for t in e]})
    # seed emergent softmax/gates + random skips into a subset of the explorer graphs (leave GPT-2 seeds 0-4 clean)
    _rng = np.random.RandomState(0)
    n_gated = 0
    for j in range(5, min(13, len(seed_graphs))):
        g2 = seed_gates_and_skips(seed_graphs[j], _rng)
        if summarize(g2)["gating"] > 0:
            seed_graphs[j] = g2; n_gated += 1
    print(f"Seeded {len(seed_graphs)} graphs; injected softmax-gates/skips into {n_gated}.", flush=True)

    front, sig2state, best_hist, msgs, all_pts = [], {}, [], [], []

    for gen in range(generations):
        print(f"\n--- Graph-Search Generation {gen+1}/{generations} ---", flush=True)
        if gen == 0:
            children = [(dict(g), None) for g in seed_graphs]
        else:
            fsorted = sorted(front, key=lambda z: z["flops"])
            prog = ask_edits(gen, front, best_hist, msgs)
            children = []
            if prog is None:
                children = [(copy.deepcopy(e["graph"]), e["sig"]) for e in fsorted[:pop_size]]
            else:
                for item in prog[:pop_size]:
                    pi = item.get("parent", -1)
                    if isinstance(pi, int) and 0 <= pi < len(fsorted):
                        par = fsorted[pi]; pg, psig = par["graph"], par["sig"]
                    else:
                        pg, psig = seed_graphs[np.random.randint(len(seed_graphs))], None
                    children.append((apply_edits(pg, item.get("ops", [])), psig))
            while len(children) < pop_size and fsorted:
                e = fsorted[len(children) % len(fsorted)]
                children.append((copy.deepcopy(e["graph"]), e["sig"]))

        results = []
        for idx, (g, psig) in enumerate(children):
            s = summarize(g); flops = active_flops(g["nodes"])
            try:
                params = params_of(g)
            except Exception as ex:
                print(f"Cand {idx+1}/{pop_size}: build error, reject ({str(ex)[:50]})", flush=True)
                results.append((g, PENALTY, flops, 0, s, "build", None)); continue
            print(f"Cand {idx+1}/{pop_size}: atoms A{s['attn']} L{s['linear']} sm{s['softmax']} "
                  f"gate{s['gating']} skip{s['skips']} | {params/1e6:.0f}M {flops/1e9:.3f}G ...", flush=True)
            if s["compute"] < MIN_COMPUTE:
                results.append((g, PENALTY, flops, params, s, "too_small", None)); continue
            try:
                import gc; gc.collect(); torch.cuda.empty_cache()
                model, loss, mem = train_and_eval_bpe_model(g["nodes"], g["edges"], d_model=DMODEL,
                                    max_iters=eval_steps, batch_size=8, block_size=BLOCK,
                                    parent_state_dict=sig2state.get(psig))
                print(f"   loss {loss:.4f}, mem {mem/1e9:.2f}GB", flush=True)
                if mem > MEM_BUDGET: results.append((g, PENALTY, flops, params, s, "memory", None))
                elif loss >= LOSS_MAX: results.append((g, loss, flops, params, s, "loss", None))
                else: results.append((g, loss, flops, params, s, "ok", model.state_dict()))
                del model; gc.collect(); torch.cuda.empty_cache()
            except Exception as ex:
                print(f"   REJECT (runtime/oom): {str(ex)[:70]}", flush=True)
                results.append((g, PENALTY, flops, params, s, "oom", None))

        for (g, l, f, p, s, r, st) in results:
            if st is not None: sig2state[graph_sig(g)] = st
        pool = [dict(graph=g, loss=l, flops=f, params=p, summary=s, sig=graph_sig(g), anchors=anchors_of(g))
                for (g, l, f, p, s, r, st) in results if l < LOSS_MAX]
        front.extend(pool)
        nd = [a for a in front if not any(
                b is not a and b["loss"] <= a["loss"] and b["flops"] <= a["flops"]
                and (b["loss"] < a["loss"] or b["flops"] < a["flops"]) for b in front)]
        seen = {}
        for a in sorted(nd, key=lambda z: z["loss"]):
            seen.setdefault(a["sig"], a)
        front = list(seen.values())
        sig2state = {k: v for k, v in sig2state.items() if k in {a["sig"] for a in front}}

        feas = sum(1 for r in results if r[5] == "ok")
        best = min((a["loss"] for a in front), default=float("nan"))
        best_hist.append(best)
        for (g, l, f, p, s, r, st) in results:
            if l < LOSS_MAX: all_pts.append((f, l))
        reasons = Counter(r[5] for r in results if r[5] != "ok")
        print(f" -> {feas}/{pop_size} feasible | front {len(front)} | best loss {best:.4f} | rejects {dict(reasons)}", flush=True)
        _save(gen, front, all_pts, sig2state)

    print("\n=== GRAPH SEARCH COMPLETE ===", flush=True)


def _save(gen, front, all_pts, sig2state):
    import glob as _glob
    for old in _glob.glob(f"{CKPT}/*.pt"):          # keep only the current front's weights on disk
        try: os.remove(old)
        except OSError: pass
    rep = []
    for i, a in enumerate(sorted(front, key=lambda z: z["flops"])):
        cfg = f"{CKPT}/graph_gen{gen+1}_ind{i+1}_config.json"
        summ = {k: v for k, v in a["summary"].items() if k != "hist"}
        wfile = None
        st = sig2state.get(a["sig"])
        if st is not None:                          # persist trained weights (reconstruct + resume)
            wfile = f"{CKPT}/graph_gen{gen+1}_ind{i+1}_loss{a['loss']:.2f}.pt"
            try: torch.save(st, wfile)
            except Exception as ex: print(f"   [warn] weight save failed: {ex}", flush=True); wfile = None
        # FULL graph is preserved: nodes (list of {id,type,kwargs}) + edges (list of [u,v])
        json.dump({"nodes": a["graph"]["nodes"], "edges": [list(e) for e in a["graph"]["edges"]],
                   "loss": a["loss"], "active_flops": a["flops"], "params": a["params"],
                   "summary": summ, "weights": wfile}, open(cfg, "w"), indent=2)
        rep.append({"rank": i + 1, "loss": a["loss"], "active_flops": a["flops"], "params": a["params"],
                    "n_nodes": summ.get("nodes"), "n_edges": summ.get("edges"),
                    "attn": summ["attn"], "linear": summ["linear"], "softmax": summ["softmax"],
                    "gating": summ["gating"], "skips": summ["skips"], "compute": summ["compute"],
                    "config": cfg, "weights": wfile})
    json.dump(rep, open(f"{CKPT}/generation_{gen+1}_report.json", "w"), indent=2)
    plt.figure(figsize=(8, 5.5))
    if all_pts:
        ap = np.array(all_pts); plt.scatter(ap[:, 0] / 1e9, ap[:, 1], c="#c3ccd6", s=24, alpha=0.6, label="all evaluated")
    fs = sorted(front, key=lambda z: z["flops"])
    if fs:
        fx = [a["flops"] / 1e9 for a in fs]; fy = [a["loss"] for a in fs]
        for a in fs:
            c = "#d97706" if a["summary"]["gating"] else ("#7c3aed" if a["summary"]["linear"] == 0 else "#2b6cb0")
            plt.scatter(a["flops"] / 1e9, a["loss"], s=110, color=c, edgecolor="white", zorder=5)
        if len(fx) > 1: plt.plot(fx, fy, "--", color="#d64545", alpha=0.7)
    plt.xlabel("Active-FLOPs / token (GFLOPs)"); plt.ylabel("Validation loss")
    plt.title(f"Atomic graph search — Gen {gen+1} Pareto front"); plt.grid(True, ls=":", alpha=0.5); plt.legend()
    plt.tight_layout(); plt.savefig(f"{CKPT}/generation_{gen+1}_pareto.png", dpi=140, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    run_graph_search(generations=100, pop_size=20, eval_steps=57860)
