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
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from rwnn.graph import RWNNGraph
from rwnn.mutator import is_valid_dag
from agentic_optimizer import claude_llm

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
    "slice":           {"start": 0, "end": 1},
    "top_k":           {"k": 2},
    "gather":          {"dim": 1},
    "scatter_add":     {"dim": 1},
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

# One-line docs per primitive. The prompt's "Primitive vocabulary" block is GENERATED from PRIMS
# (single source of truth) so it can never drift from what apply_edits actually accepts.
PRIM_DOC = {
    "linear":          "affine map; the compiler AUTO-PROJECTS mismatched dims, so wiring is forgiving",
    "layer_norm":      "normalize over the feature dim",
    "activation":      "pointwise nonlinearity (act_type: gelu|silu|relu)",
    "softmax":         "over the last dim (a router: linear(d_out=E)->softmax gives E gate weights)",
    "slice":           "take channels [start:end] of the last dim (extract one gate: slice(i,i+1))",
    "top_k":           "keep the k largest gate weights (others 0), renormalized -> sparse routing",
    "gather":          "select rows by integer indices along dim (route a token subset to an expert)",
    "scatter_add":     "add gathered/expert outputs back to their positions (recombine routed tokens)",
    "element_mul":     "elementwise product (a width-1 input broadcasts over d_model) -> apply a gate",
    "sum":             "elementwise add of ALL inputs -> residual/merge; extra inputs = skip connections",
    "concat":          "concatenate inputs along the given dim",
    "causal_attention":"one masked multi-head self-attention atom",
    "matmul":          "learned matrix multiply d_in->d_out",
    "scale_shift":     "per-channel affine (gamma*x + beta)",
    "add_bias":        "per-channel learned bias",
    "mean_reduce":     "average over the given dim",
    "dropout":         "stochastic feature zeroing (regularization)",
}

def _fmt_kwargs(kw):
    if not kw:
        return "{}"
    return "{" + ", ".join(f"{k}:{v!r}" if isinstance(v, str) else f"{k}:{v}" for k, v in kw.items()) + "}"

# Generated vocabulary block injected into SYSTEM at call time (see ask_edits).
PRIM_VOCAB = "\n".join(
    f"  {name} {_fmt_kwargs(PRIMS[name])}".ljust(46) + f"-- {PRIM_DOC.get(name, '')}".rstrip(" -")
    for name in PRIMS
)


# ----------------------------------------------------------------------------- seed encoding + trainer
# 65-D seed vector only (0 depth | 1-16 attn | 17-32 ffn | 33-48 activation | 49-64 skip). Dense only.
def _gpt2_vector(model_type):
    x = np.zeros(65); x[33:49] = 0.1  # GELU
    if model_type == "gpt2-sparse":
        x[0] = 18 / 24; x[1:17] = [1.0, 0.0] * 8; x[17:33] = [0.0, 1.0] * 8
    else:
        x[0] = {"gpt2": 6 / 24, "gpt2-medium": 18 / 24, "gpt2-large": 1.0}.get(model_type, 18 / 24)
        x[1:33] = 1.0
    return x

def vector_to_multilayer_graph(x, vocab_size=VOCAB, block_size=BLOCK, d_model=DMODEL):
    """Decode a 65-D seed vector into a dense H-DAG (attention + dense FFN + real skips). No MoE."""
    x = np.asarray(x)
    L, av, mv, cv, sv = x[0], x[1:17], x[17:33], x[33:49], x[49:65]
    n_layer = min(6 + int(L * 24), 30)
    nodes = [{"id": 0, "type": "input", "kwargs": {}},
             {"id": 1, "type": "token_embedding", "kwargs": {"vocab_size": vocab_size, "d_model": d_model}},
             {"id": 2, "type": "positional_embedding", "kwargs": {"max_seq_len": block_size, "d_model": d_model}},
             {"id": 3, "type": "sum", "kwargs": {}}]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    cx, layer_out, final_sum, active = 3, {}, {}, []
    for l in range(n_layer):
        i16 = min(int(l * 16 / n_layer), 15)
        ln1, at, sa, ln2, up, ac, dn, sm = (4 + l*10, 5 + l*10, 6 + l*10, 7 + l*10, 8 + l*10, 9 + l*10, 10 + l*10, 11 + l*10)
        if av[i16] > 0.5:
            nodes += [{"id": ln1, "type": "layer_norm", "kwargs": {"d_model": d_model}},
                      {"id": at, "type": "causal_attention", "kwargs": {"n_head": 12, "d_model": d_model, "dropout": 0.1}},
                      {"id": sa, "type": "sum", "kwargs": {}}]
            edges += [(cx, ln1), (ln1, at), (cx, sa), (at, sa)]; attn_out = sa
        else:
            attn_out = cx
        if mv[i16] > 0.5:
            act = "gelu" if cv[i16] < 0.33 else ("silu" if cv[i16] < 0.66 else "relu")
            nodes += [{"id": ln2, "type": "layer_norm", "kwargs": {"d_model": d_model}},
                      {"id": up, "type": "linear", "kwargs": {"d_in": d_model, "d_out": 4 * d_model}},
                      {"id": ac, "type": "activation", "kwargs": {"act_type": act}},
                      {"id": dn, "type": "linear", "kwargs": {"d_in": 4 * d_model, "d_out": d_model}},
                      {"id": sm, "type": "sum", "kwargs": {}}]
            edges += [(attn_out, ln2), (ln2, up), (up, ac), (ac, dn), (attn_out, sm), (dn, sm)]; mlp_out = sm
        else:
            mlp_out = attn_out
        if mv[i16] > 0.5: final_sum[l] = sm
        elif av[i16] > 0.5: final_sum[l] = sa
        if l in final_sum: active.append(l)
        cx = mlp_out; layer_out[l] = cx
    for i, l in enumerate(active):                       # real long-range skips (into a later SumNode)
        i16 = min(int(l * 16 / n_layer), 15)
        if sv[i16] > 0.5 and i + 2 < len(active):
            edges.append((layer_out[l], final_sum[active[i + 2]]))
    ln_f = 4 + n_layer * 10
    nodes += [{"id": ln_f, "type": "layer_norm", "kwargs": {"d_model": d_model}},
              {"id": 13, "type": "linear", "kwargs": {"d_in": d_model, "d_out": vocab_size}}]
    edges += [(cx, ln_f), (ln_f, 13)]
    ids = {n["id"] for n in nodes}
    return nodes, [(u, v) for u, v in edges if u in ids and v in ids]

def build_initial_population(pop_size, seed=1234):
    """5 GPT-2 seeds (incl. 2 saved elites) + (pop_size-5) diverse explorers — 65-D, dense (no MoE)."""
    rng = np.random.RandomState(seed); pop = []
    def _pad(v):
        v = np.asarray(v, dtype=float)
        return np.concatenate([v, np.zeros(65 - len(v))]) if len(v) < 65 else v[:65]
    if os.path.exists("seed_elites.json"):
        for e in json.load(open("seed_elites.json"))[:2]:
            pop.append(_pad(e["vector"]))
    pop.append(_gpt2_vector("gpt2")); pop.append(_gpt2_vector("gpt2-sparse"))
    g18 = np.zeros(65); g18[0] = 0.5; g18[1:33] = 1.0; g18[33:49] = 0.1; pop.append(g18)   # dense 18L
    while len(pop) < 5: pop.append(_gpt2_vector("gpt2-medium"))
    pop = pop[:5]
    for i in range(max(0, pop_size - 5)):
        x = np.zeros(65); x[0] = np.clip((i + 0.5) / max(1, pop_size - 5), 0.12, 0.95)
        front = rng.randint(4, 11)
        for s in range(16): x[1 + s] = np.clip((0.85 if s < front else 0.25) + rng.normal(0, 0.12), 0, 1)
        md = rng.uniform(0.35, 0.8)
        for s in range(16): x[17 + s] = np.clip((0.8 if rng.rand() < md else 0.2) + rng.normal(0, 0.1), 0, 1)
        style = i % 4
        for s in range(16):
            x[33 + s] = (rng.uniform(0, 0.30) if style == 0 else rng.uniform(0.36, 0.63) if style == 1
                         else rng.uniform(0.70, 1.0) if style == 2 else rng.uniform(0, 1.0))
        sd = rng.uniform(0.4, 0.85)
        for s in range(16): x[49 + s] = np.clip((0.8 if rng.rand() < sd else 0.2) + rng.normal(0, 0.1), 0, 1)
        for s in rng.choice(16, size=3, replace=False): x[49 + s] = rng.uniform(0.6, 0.95)
        pop.append(np.clip(x, 0, 1))
    return np.array(pop[:pop_size])

def train_and_eval_bpe_model(nodes, edges, d_model=DMODEL, max_iters=1000, batch_size=8,
                             block_size=BLOCK, parent_state_dict=None):
    """Train a compiled H-DAG on BPE tokens. Returns (model, val_loss, peak_mem_bytes)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")
    model = RWNNGraph(nodes, edges, global_d_model=d_model)
    if parent_state_dict is not None:                    # Lamarckian weight inheritance
        cs = model.state_dict(); c = 0
        for k, v in parent_state_dict.items():
            if k in cs and cs[k].shape == v.shape:
                cs[k].copy_(v); c += 1
        if c > 0: print(f" -> Inherited {c} parameter tensors from parent weights.")
    model.to(device)
    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        xb = torch.stack([torch.from_numpy((d[i:i + block_size]).astype(np.int64)) for i in ix])
        yb = torch.stack([torch.from_numpy((d[i + 1:i + block_size + 1]).astype(np.int64)) for i in ix])
        return xb.to(device), yb.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    model.train()
    for _ in range(max_iters):
        xb, yb = get_batch("train")
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(xb); loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval(); vl = []
    with torch.no_grad():
        for _ in range(5):
            X, Y = get_batch("val")
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(X); vl.append(F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)).item())
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    return model, float(np.mean(vl)), peak


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

def apply_edits(parent, ops, donors=None):
    """Apply a list of atomic edits to a copy of parent; revert entirely if the result is invalid.
    `donors` (the elite front) enables CROSSOVER via the `graft` op: copy a connected subgraph from
    another parent into this child (fresh ids, internal wiring preserved), addressable as 'graft<id>'."""
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
            elif k == "graft":                        # CROSSOVER: copy a subgraph from another parent
                j = op.get("from")
                if not donors or not isinstance(j, int) or not (0 <= j < len(donors)):
                    continue
                dmap = {n["id"]: n for n in donors[j]["nodes"]}
                sel = [nid for nid in (op.get("nodes") or []) if nid in dmap and nid not in ANCHORS]
                if not sel:
                    continue
                idmap = {}
                for did in sel:                        # copy donor nodes with fresh ids
                    nid = _rid(nodes)
                    nodes.append({"id": nid, "type": dmap[did]["type"], "kwargs": dict(dmap[did].get("kwargs", {}))})
                    idmap[did] = nid
                    label2id[f"graft{did}"] = nid       # reference a grafted node as "graft<donor_id>"
                prefix = op.get("id")
                if isinstance(prefix, str):
                    for did in sel:
                        label2id[f"{prefix}{did}"] = idmap[did]
                for (u, v) in donors[j]["edges"]:       # preserve the subgraph's internal wiring
                    if u in idmap and v in idmap:
                        edges.append((idmap[u], idmap[v]))
        except Exception:
            continue
    ids = {n["id"] for n in nodes}
    edges[:] = [(u, v) for u, v in edges if u in ids and v in ids]
    pn, pe = _prune_to_live(nodes, edges)
    if pn is None or not is_valid_dag(pn, pe) or 13 not in {n["id"] for n in pn}:
        return copy.deepcopy(parent)
    if len([n for n in pn if n["type"] in ("causal_attention", "linear", "matmul")]) < MIN_COMPUTE + 1:
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
        elif t == "causal_batch_matmul":
            f += 2 * T * DMODEL
    return f

def summarize(g):
    nodes, edges = g["nodes"], g["edges"]
    hist = Counter(n["type"] for n in nodes)
    sm = {n["id"] for n in nodes if n["type"] == "softmax"}
    sl = {n["id"] for n in nodes if n["type"] == "slice"}
    em = {n["id"] for n in nodes if n["type"] == "element_mul"}
    sl_from_sm = {v for u, v in edges if u in sm and v in sl}        # slices fed by a softmax (router)
    gate_src = sm | sl_from_sm
    gating = sum(1 for u, v in edges if u in gate_src and v in em)   # softmax/router -> element_mul gate
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


def seed_atomic_moe(g, experts, rng):
    """Seed a graph with a mixture-of-experts built ENTIRELY FROM ATOMS (no monolithic block):
    router linear(d_out=E) -> softmax -> per-expert [slice(e,e+1) * (linear->activation)] -> sum.
    This is a soft mixture (all E experts run); it lives in the gene pool and the agent can build
    the same thing from atoms. E is kept small since soft mixtures cost E x the FFN FLOPs."""
    tail, ln_f, _ = anchors_of(g)
    if tail is None or ln_f is None:
        return None
    E = int(experts); act = str(rng.choice(["gelu", "silu", "relu"]))
    ops = [{"op": "add_node", "id": "rt", "type": "linear", "kwargs": {"d_out": E}, "inputs": [tail]},
           {"op": "add_node", "id": "sm", "type": "softmax", "inputs": ["rt"]}]
    gated = []
    for e in range(E):
        ei, ai, gi, mi = f"e{e}", f"a{e}", f"g{e}", f"m{e}"
        ops += [
            {"op": "add_node", "id": ei, "type": "linear", "kwargs": {"d_out": DMODEL}, "inputs": [tail]},
            {"op": "add_node", "id": ai, "type": "activation", "kwargs": {"act_type": act}, "inputs": [ei]},
            {"op": "add_node", "id": gi, "type": "slice", "kwargs": {"start": e, "end": e + 1}, "inputs": ["sm"]},
            {"op": "add_node", "id": mi, "type": "element_mul", "inputs": [ai, gi]},
        ]
        gated.append(mi)
    ops += [{"op": "add_node", "id": "mx", "type": "sum", "inputs": [tail] + gated},
            {"op": "remove_edge", "u": tail, "v": ln_f},
            {"op": "add_edge", "u": "mx", "v": ln_f}]
    return apply_edits(g, ops)


# ----------------------------------------------------------------------------- Claude atomic-edit agent
SYSTEM = """You are Metis-Graph, an autonomous neural-architecture search agent. You evolve a
decoder-only language model whose architecture is a directed ACYCLIC graph of PRIMITIVE atoms.
You do not write the network directly — you emit a GRAPH-EDIT PROGRAM that transforms parent
graphs on the current Pareto front, using ONLY these generic atomic operations:

  {"op":"add_node","id":"<label>","type":"<primitive>","kwargs":{...},"inputs":[<node refs>]}
  {"op":"add_edge","u":<ref>,"v":<ref>}
  {"op":"remove_edge","u":<ref>,"v":<ref>}
  {"op":"remove_node","id":<real node id>}
  {"op":"graft","from":<elite index>,"nodes":[<node ids in THAT elite>]}   # CROSSOVER

A <ref> is either a real integer node id (shown to you in the context) or a "<label>" you assigned
to a node created earlier in the SAME program. `inputs` wires predecessors -> the new node.

## Crossover (combine features from parents)
`graft` copies a connected subgraph FROM ANOTHER elite ("from": its index) INTO the child you are
building — the selected `nodes` are copied with fresh ids and their internal wiring is preserved.
Each grafted node becomes addressable as "graft<original_id>" (e.g. "graft42"), so AFTER a graft you
wire it into the child with add_edge (feed its entry from the child's `tail` or a `sum`, and route its
exit into a `sum` before ln_f). This is how you recombine good motifs from two or more parents — pick
a base `parent` and graft, say, another elite's MoE/gate/attention motif onto it. Prefer combining
features from DIFFERENT elites rather than only mutating one.

## Primitive vocabulary (these are the ONLY node types; there are NO attention/FFN/MoE blocks).
## Each line is `name {default kwargs} -- description`; override any kwarg in your add_node op.
{prim_vocab}

## Fixed anchors (never remove): 0 input, 1 token_emb, 2 pos_emb, 3 emb_sum, 13 LM head.
Everything between node 3 and node 13 is yours to (re)build. For each elite you are given its
current head-feeding node (`tail`), its final layer-norm id (`ln_f`), and its residual sum-node ids.

## Emergence, not templates
Nothing is pre-built. If you want a feed-forward, wire linear->activation->linear and a `sum` for
its residual. To build a mixture-of-experts from atoms: a router linear(d_out=E)->softmax gives E
gate weights; for each expert e wire slice(e,e+1)->element_mul with that expert's output, then sum
all gated experts (+ a residual). If you want something nobody has tried, wire it.
## Your mandate (multi-objective optimization) — you ARE the optimizer, not a blind mutation operator
Two objectives are MINIMIZED, forming a Pareto front: (1) validation loss, (2) active-FLOPs/token.
Extra parallel compute costs FLOPs; skips and gates are cheap. Every generation your job is to
IMPROVE AND EXPAND the front — grow its hypervolume (dominated area) — by spending your children
across these classic multi-objective goals:
  - DOMINATE  : push an existing elite down-and-left (lower loss at the same-or-lower FLOPs).
  - EXTEND-LOSS : reach a NEW lowest-loss point (spend FLOPs where they actually buy loss).
  - EXTEND-FLOPS: reach a NEW cheapest viable point (strip FLOPs while staying under the loss cap).
  - FILL-GAP  : where two adjacent elites are far apart in FLOPs, add an intermediate trade-off.
  - DIVERSIFY : keep structurally distinct lineages alive; never collapse the whole batch onto one motif.
Budget your {pop_size} children ACROSS these goals; do not put them all on one. Learn from the last
generation's outcomes (what improved the front, what was dominated/rejected) and adapt your plan.
Reject-triggers (wasted evaluations, avoid them): peak mem >12GB, loss>=4.5, <3 compute atoms.

## Output (STRICT): first 3-4 short LEARNINGS bullets — what the last generation taught you about the
front and your plan to expand it this generation — then a fenced ```json block: a list of exactly
{pop_size} children, each
  {"parent": <elite index, or -1 for a random seed>,
   "goal": "dominate|extend-loss|extend-flops|fill-gap|diversify",
   "rationale": "<one sentence: what this child is trying to achieve on the front and why>",
   "ops":[ ...atomic ops... ]}.
An empty ops list re-trains a parent unchanged. Wire every new subgraph so it reaches node 13."""

_TSHORT = {"causal_attention": "attn", "layer_norm": "ln", "linear": "lin", "activation": "act",
           "element_mul": "emul", "softmax": "sm", "slice": "sl", "sum": "+", "gather": "gath",
           "scatter_add": "scat", "top_k": "topk", "matmul": "mm", "concat": "cat",
           "token_embedding": "tok", "positional_embedding": "pos", "input": "in"}

def build_context(gen, front, hist):
    fsorted = sorted(front, key=lambda z: z["flops"])
    lines = [f"Generation {gen}. Pareto front ({len(front)} elites), sorted by active-FLOPs:"]
    for i, e in enumerate(fsorted):
        s = e["summary"]; t, lnf, sums = e["anchors"]
        lines.append(f"  [elite {i}] loss {e['loss']:.3f}, {e['flops']/1e9:.3f}G, {e['params']/1e6:.0f}M | "
                     f"atoms: attn {s['attn']}, linear {s['linear']}, softmax {s['softmax']}, "
                     f"element_mul {s['elemmul']}, gates {s['gating']}, skips {s['skips']}")
        lines.append(f"           anchors: tail(head-input)={t}, ln_f={lnf}, sum_ids={sums[:24]}")
    # Full structure of the top elites so you can GRAFT subgraphs from them (crossover).
    lines.append("\nGraft-able structure of the leading elites (node id:type ; edges) — pick connected "
                 "subgraphs to graft with {\"op\":\"graft\",\"from\":<elite>,\"nodes\":[...]}:")
    for i, e in enumerate(fsorted[:5]):
        g = e["graph"]
        nn = " ".join(f"{n['id']}:{_TSHORT.get(n['type'], n['type'])}" for n in g["nodes"] if n["id"] not in (0, 1, 2))
        ee = " ".join(f"{u}>{v}" for u, v in g["edges"])
        lines.append(f"  elite {i} nodes: {nn}")
        lines.append(f"  elite {i} edges: {ee}")
    # front geometry to steer expansion (extremes to extend + biggest gap to fill)
    if fsorted:
        lo_loss = min(fsorted, key=lambda z: z["loss"]); li = fsorted.index(lo_loss)
        lines.append(f"\nFront geometry: lowest-loss elite = [{li}] (loss {lo_loss['loss']:.3f} @ "
                     f"{lo_loss['flops']/1e9:.3f}G) -> push it lower to EXTEND-LOSS; cheapest elite = [0] "
                     f"(loss {fsorted[0]['loss']:.3f} @ {fsorted[0]['flops']/1e9:.3f}G) -> undercut it to EXTEND-FLOPS.")
        if len(fsorted) >= 2:
            g, gi = max((fsorted[i+1]['flops'] - fsorted[i]['flops'], i) for i in range(len(fsorted)-1))
            lines.append(f"Biggest FLOP gap: between elite [{gi}] ({fsorted[gi]['flops']/1e9:.3f}G) and "
                         f"[{gi+1}] ({fsorted[gi+1]['flops']/1e9:.3f}G) -> a FILL-GAP target ({g/1e9:.3f}G wide).")
    if hist:
        lines.append("\nBest loss per generation: " + ", ".join(f"{h:.3f}" for h in hist[-8:]))
    lines.append(f"\nEmit a graph-edit program producing exactly {POP} children that EXPAND this front "
                 f"(dominate / extend-loss / extend-flops / fill-gap / diversify). Use `graft` to combine "
                 f"features from DIFFERENT elites, not just mutate one. Give each child a goal + a rationale.")
    return "\n".join(lines)

def ask_edits(gen, front, hist, msgs):
    msgs.append({"role": "user", "text": build_context(gen, front, hist)})
    prompt = "\n\n".join(f"===== {'REQUEST' if m['role']=='user' else 'YOUR RESPONSE'} =====\n{m['text']}"
                         for m in msgs[-10:])
    system = SYSTEM.replace("{prim_vocab}", PRIM_VOCAB).replace("{pop_size}", str(POP))
    text, thinking = claude_llm.invoke(system, prompt, model=os.environ["CLAUDE_MODEL"])
    if thinking:
        print(f"\n  \033[32m[THINKING] {thinking.strip()[:1200]}\033[0m", flush=True)
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    learnings = (text[:m.start()] if m else text).strip()
    if learnings:
        print(f"  \033[36m[LEARNINGS]\n  " + learnings[:1600].replace("\n", "\n  ") + "\033[0m", flush=True)
    msgs.append({"role": "assistant", "text": text})
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
    # add 2 atom-composed mixture-of-experts variants to the initial gene pool (built from atoms only)
    n_moe = 0
    for k, E in enumerate([2, 4]):
        idx = 13 + k
        if idx < len(seed_graphs):
            gm = seed_atomic_moe(seed_graphs[idx], E, _rng)
            if gm is not None and graph_sig(gm) != graph_sig(seed_graphs[idx]):
                seed_graphs[idx] = gm; n_moe += 1
    print(f"Seeded {len(seed_graphs)} graphs; injected gates/skips into {n_gated}, atom-MoE into {n_moe}.", flush=True)

    front, sig2state, best_hist, msgs, all_pts = [], {}, [], [], []

    for gen in range(generations):
        print(f"\n--- Graph-Search Generation {gen+1}/{generations} ---", flush=True)
        if gen == 0:
            children = [(dict(g), None, "seed (initial population)") for g in seed_graphs]
        else:
            fsorted = sorted(front, key=lambda z: z["flops"])
            donors = [e["graph"] for e in fsorted]     # crossover donors (graft op indexes into this)
            prog = ask_edits(gen, front, best_hist, msgs)
            children = []
            if prog is None:
                children = [(copy.deepcopy(e["graph"]), e["sig"], "reuse elite (unparseable program)")
                            for e in fsorted[:pop_size]]
            else:
                for item in prog[:pop_size]:
                    pi = item.get("parent", -1)
                    if isinstance(pi, int) and 0 <= pi < len(fsorted):
                        par = fsorted[pi]; pg, psig = par["graph"], par["sig"]; pdesc = f"elite {pi}"
                    else:
                        pg, psig = seed_graphs[np.random.randint(len(seed_graphs))], None; pdesc = "random seed"
                    goal = str(item.get("goal", "?")); rat = str(item.get("rationale", "")).strip()
                    plan = f"[{goal}] from {pdesc}: {rat}" if rat else f"[{goal}] from {pdesc}"
                    children.append((apply_edits(pg, item.get("ops", []), donors=donors), psig, plan))
            while len(children) < pop_size and fsorted:
                e = fsorted[len(children) % len(fsorted)]
                children.append((copy.deepcopy(e["graph"]), e["sig"], "pad: reuse elite"))

        results = []
        for idx, (g, psig, plan) in enumerate(children):
            print(f"Cand {idx+1}/{pop_size} PLAN {plan}", flush=True)
            s = summarize(g); flops = active_flops(g["nodes"])
            try:
                params = params_of(g)
            except Exception as ex:
                print(f"Cand {idx+1}/{pop_size}: build error, reject ({str(ex)[:50]})", flush=True)
                results.append((g, PENALTY, flops, 0, s, "build", None)); continue
            print(f"Cand {idx+1}/{pop_size}: atoms A{s['attn']} L{s['linear']} sm{s['softmax']} "
                  f"em{s['elemmul']} gate{s['gating']} skip{s['skips']} | {params/1e6:.0f}M {flops/1e9:.3f}G ...", flush=True)
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
                    "attn": summ["attn"], "linear": summ["linear"], "softmax": summ["softmax"], "gating": summ["gating"], "skips": summ["skips"],
                    "compute": summ["compute"],
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
