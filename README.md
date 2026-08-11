# Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models

Research codebase for **"Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models"**
(NeurIPS 2026 draft). It performs multi-objective **neural architecture search (NAS)** for
decoder-only language models, where each architecture is a **Heterogeneous Directed Acyclic Graph
(H-DAG)** of primitive nodes that is compiled and trained on the fly.

---

## What we are doing

An LLM agent (Metis-Agent, Gemini-backed) drives an evolutionary search over a continuous encoding
of language-model architectures. Every generation the agent **writes its own optimization code**
(PCA-EA, CMA-ES, Ridge surrogate-inverse, gap-filling, …) to propose the next population of
candidate vectors. Each candidate is decoded into a real H-DAG, compiled into a PyTorch module,
trained for one epoch on WikiText-103, and scored. A **Pareto front** is maintained across two
competing objectives.

To avoid cold-start cost, new candidates copy weight tensors in-place from their nearest
Pareto-front ancestor (**Lamarckian Weight Inheritance via continuous nearest-neighbour ancestry**),
so offspring resume rather than restart.

### Objectives (both minimized)

1. **Validation cross-entropy loss.**
2. **Active-FLOPs per token** — the *used* compute of the transformer body (attention + only the
   `top_k` active experts of each feed-forward block). Total parameter count is **not** an
   objective; only compute that actually runs is charged.

Using active-FLOPs (rather than parameter count) is deliberate: it is the objective under which
**Mixture-of-Experts (MoE)** becomes attractive — extra experts add capacity, parameters, and
memory but almost no active-FLOPs — so the search is free to discover sparse, high-capacity models.

### Hard constraints (a violator is rejected by design, not given a fake loss)

- **Peak training memory < 12 GB** (measured; OOM becomes a clean rejection). Large expert counts
  inflate memory even at low FLOPs, so this is what *bounds* MoE size on a consumer GPU.
- **Validation loss < 4.5** (rejects degenerate near-empty models).
- **At least 3 active blocks** (no embeddings-only models).

### The search space (97-D continuous encoding)

`x ∈ [0,1]^97`, decoded per layer via a 16-slot control map:

| Genes | Meaning |
|-------|---------|
| 0 | depth (6–30 layers) |
| 1–16 | attention on/off per slot |
| 17–32 | feed-forward on/off per slot |
| 33–48 | activation type (GELU / SiLU / ReLU) |
| 49–64 | genuine long-range residual **skip** (adds an earlier block's output into a later block's residual sum) |
| 65–80 | **MoE `n_experts`** per FFN slot (1 = dense, else 2/4/8) |
| 81–96 | **MoE `top_k`** per FFN slot (1 or 2 active experts) |

Width is fixed (`d_model=768`, 12 heads, 4× FFN); an FFN slot with `n_experts>1` compiles to a fused
**MoE block** (router → top-k sparse dispatch → gate-weighted combine, with a Switch-style
load-balance loss). `n_experts=1` reduces exactly to a dense GPT-2 feed-forward block.

### Initial population

20 individuals: **5 GPT-2-family seeds** (including the two best elites from the previous run) plus
**15 structurally-diverse explorers** deliberately seeded with skip connections, front-loaded
attention, non-uniform FFN placement, mixed activations, varied depth, and — for about half of them —
real MoE variance (so `n_experts`/`top_k` are not born collapsed to "dense"). A per-gene
**saturation monitor** reports any variable that freezes across the whole population for two
generations and injects diversity to escape the plateau.

---

## Layout

- `rwnn/nodes.py` — atomic node modules (embeddings, causal attention, linear, layernorm,
  activation, sum, **MoE feed-forward**, …).
- `rwnn/graph.py` — `RWNNGraph`: topologically compiles `(nodes, edges)`, auto-projects on dimension
  mismatch, executes as an `nn.Module`, and collects MoE load-balance losses.
- `rwnn/mutator.py` — DAG validity checks, structural mutations, `get_gpt2_dag()`.
- `evolve_agentic.py` — **main script**: `vector_to_multilayer_graph()` (decode), MoE-aware
  `active_flops_per_token()`, `train_and_eval_bpe_model()` (train + measure peak memory),
  `build_initial_population()`, and `run_agentic_optimization()` (the generation loop).
- `calculate_agentic_hypervolume.py` — hypervolume (S-metric) + convergence plots.
- `generate_graph_visualization.py`, `generate_all_elites_samples.py` — layouts and samples.
- `prepare_wikitext103.py` — BPE tokenize into `train.bin` / `val.bin`.
- `checkpoints/agentic-optim/` — per-generation outputs: `pareto_gen{G}_ind{I}_config.json`
  (real `nodes`/`edges`, `loss`, `active_flops`, `params`, MoE stats, `vector`), weights `.pt`,
  `generation_{G}_report.json`, and cumulative Pareto PNGs (loss vs active-FLOPs).

---

## Running

```bash
conda activate RWNNLMM
python prepare_wikitext103.py     # once: produces train.bin / val.bin
python evolve_agentic.py          # or: nohup python -u evolve_agentic.py > agentic_evolution.log 2>&1 &
```

Requires a sibling repo at `/home/stelios/repos/agentic-optimizer` (provides `MetisAgent`), a `.env`
with `GOOGLE_API_KEY`, and `train.bin` / `val.bin`. Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching to reduce fragmentation on a
12 GB card. Note: `run_agentic_optimization()` clears `checkpoints/agentic-optim/` on startup.

---

## Citation

```bibtex
@inproceedings{kyriacou2026lamarckian,
  title={Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models},
  author={Kyriacou, Stylianos},
  booktitle={Neural Information Processing Systems (NeurIPS 2026)},
  year={2026}
}
```
