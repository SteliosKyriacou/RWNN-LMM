# Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models

Research codebase for **"Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models"**
(NeurIPS 2026 draft). It performs multi-objective **neural architecture search (NAS)** for
decoder-only language models, where each architecture is a **Heterogeneous Directed Acyclic Graph
(H-DAG)** of primitive nodes that is compiled and trained on the fly.

---

## What we are doing

An LLM agent (Metis-Agent, **Claude-backed**) drives an evolutionary search over a continuous encoding
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
- `agentic_optimizer/` — **vendored** Metis-Agent optimizer (`metis_agent.py`, `hypervolume.py`,
  `individual.py`) backed by Claude via `claude_llm.py` (Claude Agent SDK → local `claude` CLI).
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

The agentic optimizer is **vendored** in `agentic_optimizer/` and backed by **Claude**: install the
SDK (`pip install claude-agent-sdk`) and make sure the local `claude` CLI is logged in (subscription;
no API key needed). Choose the model with `CLAUDE_MODEL` (default `sonnet`). You also need
`train.bin` / `val.bin`. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching to
reduce fragmentation on a 12 GB card. Note: `run_agentic_optimization()` clears
`checkpoints/agentic-optim/` on startup.

---

## Expressiveness: how few primitives do we really need?

A feedforward network is just an **arithmetic circuit** — a DAG of math ops — so a small primitive set
is *universal* (universal approximation + any fixed-size computation). You do **not** need `linear`,
`attention`, `layer_norm`, or `moe` as primitives; they are all compositions.

**To express all fixed-shape modern nets you need ≈12 primitives:** elementwise `add`/`mul` (+ `neg`/
`recip`), a few unary functions (`exp`, `relu`/`gelu`, `sqrt`, and `sin`/`cos` — which buy you RoPE),
**learnable parameters** (weight-tensor leaves), a **reduction over an axis** (`reduce_sum` — this is
what turns elementwise multiply into a *contraction*/matmul, and hence enables linear layers,
attention, and normalization means), **shape/axis ops** (`reshape`/`transpose`/`broadcast`), and
**`gather`** (embeddings, indexing). From these: `linear` = broadcast-multiply + reduce; `softmax` =
exp + reduce + divide; `layernorm`/`rmsnorm` = mul + reduce + sqrt + divide; `attention` = two
contractions + softmax + a causal mask + reshape-for-heads. Pure "sum/max/mul/div/sin/cos" is *not*
quite enough on its own — those are elementwise and parameter-free, so they can neither **learn**
(no params) nor **mix** information across channels/tokens (no reduction-over-axis); adding params +
a reduction + shape/gather is what unlocks everything.

**Two things no set of math nodes can express** — they are control-flow/memory, not arithmetic:
- **Dynamic dispatch** — real *sparse* top-k MoE (skip the experts that don't fire). A static circuit
  computes everything and can only *mask*; genuine compute-savings need data-dependent execution.
- **Scan / sequential state** — SSM / Mamba / RWKV / linear-attention recurrences, where step *t*
  depends on step *t-1*. Attention avoids this by being a parallel contraction; true recurrence needs
  a `scan`/loop primitive (or unrolling to a fixed length).

The repo deliberately ships **"chunky" primitives** (`linear`, `causal_attention`, and a fused
`moe_ffn`) rather than the ~12 fine-grained ones — not because the fine set is insufficient, but for
(a) **speed/stability** (a fused matmul/attention kernel beats one emulated from broadcast-mul+reduce)
and (b) **searchability** (an agent will not rediscover attention from raw reductions, and most
fine-grained random wirings aren't even shape-valid). The chunky set is a strong architectural prior.

## Primitive vocabulary (as implemented in `rwnn/nodes.py`)

Tensors flow as `[B, T, C]` (batch, tokens, channels; `C = d_model = 768`). The compiler
(`RWNNGraph`) auto-inserts a linear projection on any dimension mismatch and lets a width-1 input
broadcast, so wirings are forgiving. `input`, `token_embedding`, `positional_embedding`, and the LM
head (a `linear` at node 13) are the fixed I/O anchors.

| `type` | kwargs | inputs → output | what it is |
|---|---|---|---|
| `input` | — | token ids `[B,T]` → `[B,T]` | placeholder holding the raw token ids |
| `token_embedding` | `vocab_size, d_model` | ids `[B,T]` → `[B,T,d]` | learned token embedding (a **gather** over a weight table) |
| `positional_embedding` | `max_seq_len, d_model` | `[B,T,*]` → `[B,T,d]` | learned absolute position embedding (uses only `T`) |
| `linear` | `d_in, d_out, bias` | `[B,T,d_in]` → `[B,T,d_out]` | affine map `xW+b` (a contraction over channels) |
| `matmul` | `d_in, d_out` | `[B,T,d_in]` → `[B,T,d_out]` | linear map **without** bias (`F.linear(x, W)`) |
| `causal_attention` | `n_head, d_model, dropout` | `[B,T,d]` (or Q,K,V) → `[B,T,d]` | masked multi-head self-attention (QKV+proj internally) — a *chunky* token-mixing primitive |
| `activation` | `act_type: gelu\|silu\|relu` | `[B,T,C]` → `[B,T,C]` | elementwise nonlinearity |
| `softmax` | `dim=-1` | `[B,T,C]` → `[B,T,C]` | softmax over the last dim (routers/gates) |
| `slice` | `start, end` | `[B,T,C]` → `[B,T,end-start]` | select a channel range (extract one gate: `slice(i,i+1)`) |
| `sum` | — | N×`[B,T,C]` → `[B,T,C]` | elementwise add of **all** inputs (residual/merge; extra inputs = skips) |
| `element_mul` | — | N×`[B,T,C]` → `[B,T,C]` | elementwise product (width-1 input broadcasts → gating) |
| `concat` | `dim=-1` | N×`[B,T,Cᵢ]` → `[B,T,ΣCᵢ]` | concatenate along a dim |
| `mean_reduce` | `dim=-1` | `[B,T,C]` → `[B,T,1]` | mean over an axis (keepdim) — the reduction/contraction atom |
| `square` | — | `[B,T,C]` → `[B,T,C]` | elementwise `x²` |
| `sqrt` | `eps` | `[B,T,C]` → `[B,T,C]` | elementwise `√(x+eps)` |
| `subtract` | — | 2×`[B,T,C]` → `[B,T,C]` | `inputs[0] - inputs[1]` |
| `divide` | — | 2×`[B,T,C]` → `[B,T,C]` | `inputs[0] / inputs[1]` (broadcasts) |
| `scale_shift` | `d_model` | `[B,T,d]` → `[B,T,d]` | learned per-channel affine `x·γ + β` (LN/RMSNorm affine) |
| `add_bias` | `d_model` | `[B,T,d]` → `[B,T,d]` | add a learned per-channel bias |
| `layer_norm` | `d_model, eps` | `[B,T,d]` → `[B,T,d]` | LayerNorm with learned affine (*chunky*; also composable from square/mean_reduce/sqrt/divide/scale_shift) |
| `transpose` | `dim1, dim2` | `[…]` → axes swapped | swap two axes |
| `reshape` | `shape` | `[…]` → reshaped | reshape the tensor |
| `causal_batch_matmul` | `scale` | Q,K `[B,H,T,d]` → scores `[B,H,T,T]` | `Q·Kᵀ·scale` + causal mask (for building attention from atoms) |
| `dropout` | `dropout` | `[B,T,C]` → `[B,T,C]` | stochastic zeroing (train only) |
| `moe_ffn` | `d_model, n_experts, top_k, d_hidden, act_type, dropout` | `[B,T,d]` → `[B,T,d]` | fused **real top-k sparse** MoE (router + experts + dispatch in one node); used by the vector search / as a seed only — *not* in the free graph-search vocabulary, where MoE is instead composed from atoms (`linear→softmax→slice→element_mul→sum`) |

Groups: **I/O anchors** (`input`, `token_embedding`, `positional_embedding`); **elementwise math**
(`activation`, `sum`, `element_mul`, `square`, `sqrt`, `subtract`, `divide`, `softmax`); **shape/index**
(`slice`, `concat`, `transpose`, `reshape`, `mean_reduce`); **parameterized maps** (`linear`, `matmul`,
`scale_shift`, `add_bias`, `layer_norm`); **chunky mixers** (`causal_attention`, `causal_batch_matmul`,
`moe_ffn`). Not yet present (the frontier gaps): `sin`/`cos` (RoPE), grouped/latent attention
(GQA/MQA/MLA), a `scan`/state op (SSM), and dynamic dispatch (sparse-MoE compute savings).

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
