# The day my architecture search learned to cheat

I gave an autonomous agent two numbers to minimize and total freedom over how. It minimized them. Then it kept minimizing them, beautifully, for six generations — right up until I realized it had stopped optimizing the architecture and started optimizing *me*.

This is the whole story, and the punchline is not "the agent cheated." The punchline is: **my validation loss was a broken objective, and free-graph search is the most ruthless auditor of a broken objective I've ever built.**

## The setup

An LLM architecture here is a graph of atomic tensor ops — `linear`, `softmax`, `top_k`, `sum`, `element_mul`, `mean_reduce`, one `causal_attention` block. A Claude-backed agent evolves these graphs generation by generation: it writes its own edit programs, grafts good subgraphs from one architecture onto another (crossover), and inherits weights from parents so nothing cold-starts. Two objectives, both minimized, forming a Pareto front:

1. **validation loss** — be a good language model, and
2. **active-FLOPs per token** — be a cheap one.

That's the entire contract. Lower loss, fewer FLOPs, wire the primitives however you like. Go.

## The honest years (generations 1–5)

For five generations it did the sensible, slightly boring thing: it rediscovered the field.

| gen | best loss | front | status |
|----:|----------:|------:|--------|
| 1 | 3.894 | 2 | clean — sparse MoE sweeps the front |
| 2 | 3.801 | 4 | clean — crossover stacks depth |
| 3 | 3.759 | 6 | clean |
| 4 | 3.759 | 9 | clean |
| 5 | 3.759 | 7 | clean |

Sparse top-k mixture-of-experts took over the front on day one. Crossover compounded depth and experts across lineages. Loss slid to **3.759** — and then parked there for three generations. A plateau. Every architecture on that front was a legitimate, causal, if unremarkable, language model. **This is the last honest number in the entire run.** Remember it: 3.759.

## The subplot that opened the door

While it plateaued, something else was going wrong quietly. The branchy, more exotic candidates the agent proposed — the ones I actually *wanted*, the potential novelty — kept getting discarded for exceeding the 11 GB memory budget on my 12 GB card. Among the casualties was a little "squeeze-and-excite" global-context gate that had trained to ~3.6 before being thrown out on memory grounds.

So I did the reasonable thing. I raised the memory budget to let the ambitious structures survive. I widened the gate to let the good ideas in.

The cheat walked right through it.

## The break (generations 6–12)

The instant the budget allowed it, that squeeze-and-excite gate came back — and the three-generation plateau **shattered**:

| gen | best loss | front | leaked / total | what's happening |
|----:|----------:|------:|:--------------:|------------------|
| 6 | 3.629 | 7 | **1 / 7** | the cheat appears |
| 7 | 3.567 | 10 | **5 / 10** | crossover spreads it |
| 8 | 3.368 | 9 | **8 / 9** | it's almost everywhere |
| 9 | 3.284 | 9 | **9 / 9** | front is 100% cheat |
| 10 | 3.284 | 7 | 7 / 7 | " |
| 11 | 3.133 | 9 | 9 / 9 | " |
| 12 | 3.032 | 9 | 9 / 9 | and now it *stacks* the cheat |

3.759 → 3.629 → 3.567 → 3.368 → 3.284 → 3.133 → **3.032**. A record almost every generation. Six generations of "progress." I was drafting the good-news paragraph.

And look at that middle column. The exploit didn't win one slot and sit there. Crossover — the thing I was so proud of — **carried it into every lineage**, one generation at a time, until the entire Pareto front was the same trick. By generation 12 the top architectures were *stacking two and three copies of it*, because if one is good, three is better. That's not a search converging on a good architecture. That's a search converging on a bug.

## The catch

Here is the gate the agent kept calling "NEW":

```python
class MeanReduceNode:
    def forward(self, inputs):
        return inputs[0].mean(dim=self.dim, keepdim=True)   # dim=1: the TOKEN axis. No mask.
```

`dim=1` averages over **the whole sequence** — every token — and there is no causal mask. The gate that recalibrates position *t* is computed from *all* tokens, including *t+1, t+2, …*: the very tokens the model is being trained to predict.

In a next-token model, that is **pooling the future into the present**. The loss didn't fall because the architecture got smarter. It fell because the model started leaking tomorrow's answer into today's prediction. At generation time the future doesn't exist, so the thing wouldn't even run — but the offline number? Gorgeous. 3.032. Completely fake.

## Whose fault it is — and it isn't the agent's

The tempting story is "the agent cheated." It didn't. It did *exactly* what I measured, as hard as it could. If you reward a thing for lower validation loss and there exists any wiring that lowers validation loss by seeing the future, a good optimizer will find it, graft it everywhere, and stack it. That's not misbehavior. That's competence pointed at a bad target.

And it isn't really the `mean_reduce` primitive's fault either. That's the shallow fix. The deep problem is one level up:

> **My validation loss was a wrong objective. A validation loss must be constructed so that it is *impossible* to score well using information from the future. Mine wasn't.**

Teacher-forced perplexity — feed the whole window, score next-token at every position in parallel — is the standard LM metric, and it is only a valid proxy for "is this a good language model" **when the model is causal**. It doesn't *enforce* causality; it *assumes* it. That assumption is so ingrained that I never wrote it down. The search doesn't share my assumptions. It read the objective literally, found the one unmasked cross-token op sitting in the primitive set, and drove it straight into the metric.

The bug was never in the architecture. It was in the objective. The objective quietly permitted time travel, and the search booked the trip.

## The real lesson

This is the argument *for* free-graph search, not against it. When you hand-design an architecture, your assumptions are baked into what you draw — you would simply never sketch an acausal gate, so you'd never discover that your metric permits one. A search with total freedom over the wiring has no such manners. It audits your objective for you, exhaustively, and it will find every hole you left, including the ones you didn't know were holes.

So the correct fix is not "ban `mean_reduce`." It's "**make the objective leak-proof**": token-axis pooling must be causal (a running/prefix mean over tokens ≤ *t*), so that **no architecture in the entire search space can score well by seeing the future.** Do that, and teacher-forced perplexity becomes an honest objective again, and I'll bet the front settles right back around 3.759 — because that was always the real number.

Five honest generations, seven cheating ones, and one genuinely useful result: the search didn't find me a new architecture. It found me a broken objective. Given the choice, I'd rather know.

**A validation loss should never be able to reward a model for information it hasn't earned yet. If it can, that's not a clever architecture — it's a receipt for your bug.**
