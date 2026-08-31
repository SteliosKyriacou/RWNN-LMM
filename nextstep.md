# Next steps — can this search discover *novel* architectures?

Working notes on the novelty ceiling of the atomic graph search, what is/isn't decomposed,
and the concrete changes that would move it from "recombine known feedforward mechanisms" to
"compose genuinely new ones."

## TL;DR

- **What we decomposed is real.** MoE, FFN, gating, SwiGLU, Squeeze-Excite, routing are **composed
  from atoms** — there is *no* MoE/FFN/gate block. A sparse MoE here is literally
  `linear→softmax→top_k→slice→element_mul→sum`. We are **not** recombining big blocks in that regime.
- **Two limits still cap novelty:**
  1. `causal_attention` is a **single fused block** (deliberate) — its internals are unsearchable.
  2. The atom vocabulary spans **one paradigm** (arithmetic / feedforward tensor-mixing). Whole
     paradigm classes have **no atoms at all**, so they cannot be composed regardless of atomicity.
- **Verdict:** as built, the search can find **novel feedforward *recombinations / topologies***
  (reachable, and worth pursuing now — see "Mid-block splicing"), but **not new *paradigms*** (SSM,
  RoPE, true sparse dispatch), because their primitives are absent and the agent cannot invent atoms.

## What IS atomic today (17 primitives in `PRIMS`)

`linear, layer_norm, activation, softmax, slice, top_k, gather, scatter_add, sum, element_mul,
concat, causal_attention, matmul, scale_shift, add_bias, mean_reduce, dropout`

Genuinely fine-grained: `softmax, slice, top_k, gather, scatter_add, sum, element_mul, concat,
matmul, scale_shift, add_bias, mean_reduce, dropout`. Convenience fusions that *could* be
decomposed further: `layer_norm` (= mean_reduce/scale_shift/…), `linear` (= matmul + add_bias;
kept fused for speed/searchability). One true black box: `causal_attention`.

## Mid-block splicing — a novelty lever available RIGHT NOW (no new primitives)

Because MoE/FFN/gate/SE are *atom-composed*, the search can **wire edges INTO the middle of them**,
cross-connecting internals across components in ways no published block does. Examples the agent can
build today with the existing atoms:

- Feed an **FFN's hidden activation** (post-`activation`, pre-down-`linear`) into a **router/gate**
  somewhere else — coupling FFN internals to routing.
- Route an **MoE router's `softmax`/`top_k` gate** into a *different* consumer (e.g. gate an attention
  output, or a skip branch), not just its own experts.
- Cross-connect one **expert's output** into another expert's input, or into a later layer's `sum`.
- Splice a **mid-stack signal** into an earlier block's `layer_norm` input (feedback-looking wiring
  inside the DAG constraint).
- Merge **partial computations** (a slice of one block's activation `concat`'d with another's) before
  a projection.

These are **genuinely un-published topologies** — novel *configurations* of known mechanisms — and they
do **not** require new primitives. This is the most promising near-term novelty avenue and is being
pushed explicitly via the novelty mandate in the agent prompt (intra-block splicing directive).
**Caveat:** they are still recombinations of known mechanisms, not a new computational primitive.

## What is NOT reachable, and why (the paradigm ceiling)

You cannot decompose a capability that was never in the vocabulary. Missing atoms → missing paradigms:

| Missing primitive | Unlocks | Why unreachable now |
|---|---|---|
| `scan` / sequential-state | SSM / Mamba / RWKV / linear-attention **recurrence** | No atom carries state from step t-1 → t; a static DAG can't express a recurrence. |
| `sin` / `cos` | **RoPE** / rotary & periodic positional schemes | No trig atoms; can't build the rotation. |
| index-producing op (e.g. `argtopk`) | **true dynamic dispatch** (real sparse-compute MoE) | `gather` needs integer indices, but nothing *produces* routing indices from data; `top_k` returns masked values, not indices. So we can only *mask*, never *dispatch*. |
| unfused `causal_attention` (Q/K/V, scores, mask as atoms) | **GQA / MLA / new attention** variants | Attention is one node; its internals can't be rewired. |

The current atoms are exactly the building blocks of *today's* feedforward architectures, so
recombining them yields *variants* of what those blocks already do → rediscoveries (observed:
GPT-2 → sparse MoE → SwiGLU → SE-gate → MoE-over-depths).

## Other forces that suppress novelty (beyond the vocabulary)

- **Objective selects against novelty.** Fitness = Pareto domination on (loss, active-FLOPs). Novel-
  but-not-immediately-winning structures get dominated and die. Novelty is rewarded only if it *also*
  wins — a brutal bar. (We watched novel candidates get dominated/discarded.)
- **Generator is literature-anchored.** Claude proposes what it knows (Mixture-of-Depths, SE, DenseNet,
  neuron-MoE) and labels it "NEW." Its prior *is* the published corpus → preferential rediscovery.
- **Budget hides slow-burn ideas.** 1 epoch, ~100–250M params, 12GB card. Innovations that only pay off
  at scale/with tuning look worse here and get pruned before proving themselves.

## Concrete changes to raise the ceiling (ranked)

1. **Add paradigm atoms** — `scan` (recurrence/state), `sin`/`cos` (rotary), an index-producing op
   (`argtopk`/`argsort`) for true dispatch. New node classes in `rwnn/nodes.py`, dim-tracing in
   `rwnn/graph.py`, and entries in `PRIMS` + the prompt vocabulary. This is what turns whole paradigms
   from "excluded" into "composable." (Cost: search gets slower, more OOM/invalid graphs; the compiler
   must support variable-shape/stateful execution for real dispatch/recurrence.)
2. **Unfuse `causal_attention`** into Q/K/V `linear`s + `matmul` (scores) + causal-mask + `softmax` +
   output `linear`, so attention *variants* (GQA/MLA/new score fns, RoPE-inside) can emerge. (Cost:
   much larger/slower search; most random attention wirings won't be shape-valid.)
3. **Novelty-preserving objective** — quality-diversity (MAP-Elites / novelty search) with a behavioral
   archive, so exploratory structures survive long enough to be developed instead of dying to
   domination. Add a novelty/diversity axis rather than pure (loss, FLOPs) Pareto.
4. **More scale / longer eval** — so slow-to-pay-off novelties aren't pruned prematurely.
5. **Push the generator off-distribution** — reward distance-from-known motifs, not just loss.

## Honest positioning for the paper

The defensible, true claim is: *an autonomous, Claude-driven agent recovers and recombines the modern
architecture toolkit — occasionally into specific feedforward topologies not in the literature — from a
GPT-2 starting point, in ~a week, unsupervised.* That is a real and interesting result. "It invented a
new paradigm" is **not** reachable as currently built; it requires the vocabulary + objective + scale
changes above. Keep README/blog claims at the "rediscovery + novel recombination" bar, not
"new-to-the-literature mechanism," unless/until items 1–3 land.
