# CLAUDE.md

Guidance for working in this repo. This is a research codebase for the paper
**"Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models"** (NeurIPS 2026 draft).

## What this project does

Neural architecture search (NAS) for decoder-only LMs. An architecture is a **Heterogeneous
Directed Acyclic Graph (H-DAG)** of primitive nodes, compiled and trained on the fly. An
LLM agent (Metis-Agent, Gemini-backed) drives a multi-objective search that writes its own
optimization code each generation. Two objectives are minimized on a Pareto front:
**validation cross-entropy loss** and **trainable parameter count**. New candidates copy weights
from their nearest Pareto-front parent ("Lamarckian inheritance") to skip cold-start training.

## Layout

- `rwnn/nodes.py` — atomic node modules (embeddings, causal attention, linear, layernorm, activation, sum, …).
- `rwnn/graph.py` — `RWNNGraph`: topologically sorts `(nodes, edges)`, auto-inserts `EdgeConnection`
  projections on dimension mismatch, executes as an `nn.Module`. **The LM head is conventionally node 13**;
  `forward` returns node 13's output if present.
- `rwnn/mutator.py` — DAG validity checks, structural mutations, and `get_gpt2_dag()` (GPT-2 as an H-DAG).
- `evolve_agentic.py` — **main script**. `vector_to_multilayer_graph()` decodes a 65-D vector → graph;
  `train_and_eval_bpe_model()` trains one candidate; `run_agentic_optimization()` is the generation loop.
- `calculate_agentic_hypervolume.py` — hypervolume (S-metric) + convergence plots → `assets/`.
- `generate_graph_visualization.py`, `generate_all_elites_samples.py` — NetworkX layouts and autoregressive samples.
- `prepare_wikitext103.py` / `prepare_openwebtext.py` — BPE tokenize into `train.bin` / `val.bin`.
- `checkpoints/agentic-optim/` — per-generation outputs: `pareto_gen{G}_ind{I}_config.json` (real `nodes`/`edges`,
  `params`, `loss`, `vector`), `..._loss{L}.pt` weights, `generation_{G}_report.json`, Pareto PNGs.

## The 65-D encoding (read this before analyzing architectures)

`x ∈ [0,1]^65`: dim 0 = depth (maps to 6–30 layers); 1–16 = per-layer attention on/off (>0.5);
17–32 = per-layer MLP on/off; 33–48 = activation type; 49–64 = cross-layer skip bit.
Per-layer control is **aliased**: layer `l` reads slot `idx = int(l*16/n_layer)`, so adjacent layers
can share a gene. **Width is fixed** (`d_model=768`, 12 heads, 4× MLP) — only depth and per-layer
block placement are searched.

**Analyze the compiled graph, not the vector.** The vector's skip bit does not equal a real skip:
the decoder only keeps a skip edge if the target layer's node survives defensive edge filtering.
Always parse `nodes`/`edges` from the config JSON. Node id → layer = `(id-4)//10`; role = `(id-4)%10`
(0 ln1, 1 attn, 2 attn_sum, 3 ln2, 4 mlp_up, 5 act, 6 mlp_down, 7 mlp_sum). ids 0–3 = input/embeddings.

### Known finding (verified from the compiled graphs)
The elites' apparent "long skip highways" are **not** novel long-range residuals. Each decomposes into
the two edges of one ordinary pre-norm residual block (one → LN, one → residual `sum`), whose input is
the *immediately preceding active block* — it merely spans depth slots the search left empty. No skip
jumps over live computation. Treat README/blog claims of "multi-hop skip residual streams" with skepticism.

## Running

```bash
conda activate RWNNLMM
python evolve_agentic.py          # or: nohup python -u evolve_agentic.py > agentic_evolution.log 2>&1 &
```

Requires:
- A sibling repo at **`/home/stelios/repos/agentic-optimizer`** (hardcoded `sys.path` in `evolve_agentic.py:29`)
  providing `agentic_optimizer.metis_agent.MetisAgent`. Not part of this repo.
- `.env` with `GOOGLE_API_KEY` (+ optional `GEMINI_MODEL`). Loaded manually at the top of `evolve_agentic.py`.
- `train.bin` / `val.bin` (git-ignored; produced by `prepare_wikitext103.py`).

## Gotchas / conventions

- **Long-running.** Each candidate trains `eval_steps` (currently 57860 ≈ 1 epoch); ~1h/candidate,
  ~7–8h/generation, 100 generations planned. Check `ps`/`nvidia-smi` before assuming a run is live.
- **`run_agentic_optimization()` wipes `checkpoints/agentic-optim/` on startup** (`shutil.rmtree`).
  Back up prior runs first.
- **CUDA OOM is swallowed**: a failed candidate is caught and scored `val_loss=99.9` (lost data point).
  On the 12 GB card, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching.
- Loss constraint is hard: any candidate with `val_loss >= 6.0` is rejected (set to 99.9, weights dropped).
- No weight tying — token embedding and LM head are separate; ~77M params live in embeddings+head at `d_model=768`.
- `.gitignore` excludes `*.bin`, `*.pt`, `checkpoints/`, `.env`.

## Git

Default branch is `main`. Work happens on feature branches (`scale-up`, `fix`, …). Commit/push only when asked.
