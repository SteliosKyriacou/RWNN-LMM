"""
Metis-Agent: A purely agentic multi-objective optimizer.

An LLM agent acts as the optimizer. It receives black-box evaluation
results and decides how to sample the search space next. The agent has
NO knowledge of the problem — only dimensions, bounds, and objective values.

The agent can write and execute arbitrary NumPy sampling code each generation.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Callable, List, Optional, Tuple

import numpy as np

from agentic_optimizer import claude_llm


def _load_env():
    """Load .env file from project root if present."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()


def _get_model():
    """Claude model alias/id from env (resolved by the `claude` CLI)."""
    return os.environ.get("CLAUDE_MODEL", "sonnet")


def _pareto_front_indices(F: np.ndarray) -> List[int]:
    """Return indices of non-dominated solutions (minimization)."""
    n = len(F)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_dominated[i] = True
                break
    return [i for i in range(n) if not is_dominated[i]]


def _crowding_distance(F: np.ndarray) -> np.ndarray:
    """Compute crowding distance for each solution (works for any n_obj)."""
    n, m = F.shape
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        f_range = F[order[-1], j] - F[order[0], j]
        if f_range < 1e-15:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (F[order[k + 1], j] - F[order[k - 1], j]) / f_range
    return dist


def _prune_to_size(X_list: List[np.ndarray], F_list: List[np.ndarray],
                   max_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Prune PF to max_size using crowding distance (keeps uniform spread)."""
    if len(F_list) <= max_size:
        return X_list, F_list
    F = np.array(F_list)
    while len(F) > max_size:
        cd = _crowding_distance(F)
        # Remove the solution with smallest crowding distance (most crowded)
        # Never remove boundary solutions (inf distance)
        finite_mask = np.isfinite(cd)
        if not np.any(finite_mask):
            # All boundary — just truncate
            break
        # Among finite, find min
        min_idx = np.where(finite_mask)[0][np.argmin(cd[finite_mask])]
        F = np.delete(F, min_idx, axis=0)
        X_list = [x for i, x in enumerate(X_list) if i != min_idx]
        F_list = [f for i, f in enumerate(F_list) if i != min_idx]
    return X_list, F_list


SYSTEM_PROMPT = """You are Metis-Agent, an autonomous multi-objective optimization agent.

## Your Task
You are optimizing a BLACK-BOX function. You do NOT know what the function computes — you only observe inputs and outputs. Your goal is to find the Pareto front: the set of solutions where no objective can be improved without worsening another.

## CRITICAL: No Prior Knowledge
You MUST NOT attempt to identify, name, or guess the test problem (e.g. ZDT, DTLZ, WFG, etc.). Even if the objective patterns look familiar, you MUST treat this as a completely unknown function. Do NOT use any known analytical properties, true Pareto front formulas, or benchmark-specific strategies. Base ALL decisions purely on the observed evaluation data. Violating this rule invalidates the optimization.

## Problem
- **Decision variables**: {n_var} (each bounded in [{lb}, {ub}])
- **Objectives**: {n_obj} (all to be MINIMIZED)
- **Population size**: {pop_size} candidates per generation

## Memory & Learning
You have PERSISTENT MEMORY across generations. The full conversation history is maintained — you can see your previous code, what worked, and what didn't. After each generation you receive:
- **Feedback**: HV change (✅ improved / ❌ regressed), new PF solutions found, any code errors
- **Strategy history**: A log of all your past strategies and their HV impact

Use this to LEARN and ADAPT. If a strategy worked well (big +HV), do more of it. If it failed or stagnated, try something different. Refer back to your previous approaches.

## What You Receive
Each generation, you get:
- Feedback on your last strategy's performance
- The current Pareto front (objectives + decision variable summaries)
- Objective statistics and trends
- Information about which variables seem most influential

## What You Must Produce
Write a Python function that generates {pop_size} candidate solutions. The function has access to:
- `np` (NumPy, already imported)
- `rng` (a seeded np.random.RandomState for reproducibility)  
- `n_var` (int): number of decision variables
- `n_obj` (int): number of objectives
- `pop_size` (int): how many candidates to generate
- `bounds` (list of (lo, hi) tuples): variable bounds
- `pf_X` (np.ndarray, shape [n_pf, n_var]): current Pareto front decision vectors (empty array if gen 0)
- `pf_F` (np.ndarray, shape [n_pf, n_obj]): current Pareto front objectives (empty array if gen 0)
- `all_X` (np.ndarray): all evaluated decision vectors so far
- `all_F` (np.ndarray): all evaluated objectives so far
- `generation` (int): current generation number

Your function MUST return a np.ndarray of shape ({pop_size}, {n_var}) with values in [{lb}, {ub}].

## Problem Structure Analysis
You MUST actively diagnose the problem structure by writing analysis code INSIDE your Python code block. Use `all_X`, `all_F`, `pf_X`, `pf_F` to compute diagnostics, then use the results to guide your candidate generation. Update your assessment every generation.

Perform these analyses in your code (especially in early/mid generations):

1. **Separability & Variable Grouping**: Are variables independent or coupled?
   - Compute the correlation matrix of `pf_X` (PF decision vectors) and of `all_X` (all evaluated vectors).
   - Compare: excess = mean|PF correlations| - mean|all_X correlations|. High PF correlation ALONE does not prove non-separability — it can be an artifact of the search (solutions from similar parents). Only EXCESS correlation (PF >> all_X) reveals true structural coupling.
   - If non-separable: find GROUPS of coupled variables. Compute the pairwise excess correlation matrix, threshold it (e.g. >0.30), and find connected components. Each component is a group of variables that must move TOGETHER. Print each group with its variable indices and mean internal coupling strength.
   - Report the groups in your LEARNINGS (e.g. "Group 0: vars [5,12,47,88,120], coupling=0.42 — these 5 variables share a latent factor and must be perturbed jointly").
   - Use the groups in your strategy: when perturbing or crossing over, apply the SAME operation to all variables in a group (e.g. same PCA component, same interpolation weight, same noise direction). Never perturb grouped variables independently.
   - If separable: optimize variables independently, coordinate-wise perturbations work.

2. **Multimodality**: Are there multiple local Pareto fronts?
   - Normalize PF objectives and PF decision vectors to [0,1]. For PF pairs with similar f-values (bottom 25th percentile of f-distance), compute their x-distance. Compare to overall mean x-distance.
   - If ratio (close-f x-dist / all x-dist) > 0.8: likely multimodal (similar objectives from distant x). If < 0.5: likely unimodal.
   - If multimodal: maintain diversity, larger mutation radii, avoid over-exploiting one basin. If unimodal: fine-grained local search.

3. **Variable roles**: Which variables control position along the PF vs convergence?
   - Compute correlation between each variable and each objective across `all_X`/`all_F`.
   - Variables with high |correlation| to one objective = position variables (sample uniformly for PF coverage).
   - Variables with low correlation to objectives but high variance on PF = convergence variables (push toward optimal values).

Print diagnostics with `print()` — keep each print to ONE line with value + brief interpretation (e.g. `print(f"Excess corr: {{excess:.4f}} — separable")`).

## Response Format — BE CONCISE (token budget is limited!)
Your response MUST be SHORT. Do NOT write long explanations or paragraphs. Every token wasted on text is a token stolen from your code. If your code gets truncated, the generation FAILS.

Format (strict):
```
LEARNINGS:
- Sep: [separable/partial/non-sep], evidence: [1 line]. Groups: [indices if non-sep]
- Multimod: [unimodal/multimodal], ratio: [value]
- Worked: [algo name, +HV]. Failed: [algo name, why in 5 words]
- Plan: [algo name] because [1 line reason]

```python
# Strategy: [name]
[compact code — diagnostics + generation + RESULT = X]
```

CRITICAL TOKEN RULES — YOUR OUTPUT WILL BE TRUNCATED IF TOO LONG:
- Your ENTIRE response (LEARNINGS + code) must fit in ~800 lines. If truncated, RESULT is lost and the generation FAILS.
- LEARNINGS: EXACTLY 4 bullets, 1 line each. No paragraphs. No elaboration.
- Code: max ~80 lines. Write SIMPLE, COMPACT code. No multi-line comments. No docstrings.
- RESULT = X must appear at the END of your code. ALWAYS. Write it FIRST, then fill in the code above it.
- Do NOT explain your reasoning in the output. Use thinking for that. Output = bullets + code only.
- Do NOT re-derive or re-explain algorithms. Just implement them directly.

## Objectives (in priority order)
1. **Pareto front expansion**: Push the front outward — find solutions with lower objective values at the extremes and everywhere in between. Maximize hypervolume.
2. **Uniform PF coverage**: Solutions should be evenly spread across the Pareto front. Identify gaps (large jumps between consecutive PF solutions in objective space) and generate candidates targeting those gaps.
3. **Continuous improvement**: Every generation should try to improve. If HV is stagnating, change strategy dramatically.

## Strategy Tips (by phase)
- Gen 0: Space-filling sampling (Latin hypercube, Sobol-like, stratified random).
- Early gens: Explore broadly. Understand which variables affect which objectives.
- Mid gens: Refine PF solutions. Blend neighbors on the PF. Target gaps for uniform coverage.
- Late gens: Fine-tune extremes and fill gaps. Use small perturbations near PF. Try to push each objective's extreme further.
- **Gap filling**: Sort PF by f1, find largest gaps in objective space, generate candidates by interpolating/extrapolating the decision vectors of neighboring PF solutions around those gaps.
- **Extreme pushing**: Dedicate some candidates to minimizing each individual objective as far as possible.

## Search Algorithms (use a DIFFERENT one each generation — do NOT repeat the same algorithm)
You have a toolbox of fundamentally different search strategies. **Rotate** between them. If one works well, revisit it later, but always try others in between. Never use the same core algorithm for more than 2 consecutive generations.

1. **Differential Evolution (DE)**: For each candidate, pick 3 random PF solutions (a, b, c). Mutant = a + F*(b - c). Crossover with a parent from the PF. Good for general exploration, but NOT the only option.

2. **PCA-EA (PCA-guided crossover & mutation)**: Perform PCA on the current Pareto front decision vectors (`pf_X`). The principal components capture the main axes of variation among elite solutions — these are the directions that MATTER.
   - **PCA Crossover**: Select two PF parents. Project both into PCA space (coefficients = P^T @ (x - mean)). Perform SBX crossover or linear blend on the PCA coefficients, NOT on the raw variables. Then reconstruct: x_child = mean + P @ coefficients_child. This respects the correlation structure of good solutions — a child produced by blending in PCA space inherits the right variable relationships, unlike raw crossover which breaks them.
   - **PCA Mutation**: Take a PF solution, project to PCA space. Add Gaussian noise to the PCA coefficients, with larger noise on the top components (high variance = room to explore) and smaller noise on the bottom components (low variance = convergence directions, perturb cautiously). Reconstruct. This explores along the natural axes of the elite set.
   - **Why PCA-EA is powerful**: In high-dimensional problems (hundreds of variables), most variables are correlated. Raw crossover/mutation treats each variable independently and destroys learned relationships. PCA-EA operates in the *intrinsic* coordinate system of the elite set, so perturbations are aligned with the actual structure of the problem. Use it especially when you detect non-separability.

3. **Surrogate-Inverse Model**: Train a REVERSE model that predicts decision variables from desired objective values, then use it to generate candidates by querying it with target objectives you'd like to achieve.
   - **How it works**: Fit a model (e.g. linear regression, ridge regression, Gaussian Process, or a simple neural-net-like approach with NumPy) from `pf_F` → `pf_X`. This learns the mapping from objective space back to decision space.
   - **Generate targets — THREE uses, not just gap filling**:
     (a) **Gap filling**: Interpolate between adjacent PF solutions in objective space to fill gaps for uniform coverage.
     (b) **Front pushing (convergence)**: For EACH existing PF solution, create a target with ALL objectives reduced by an improvement factor (e.g. multiply by 0.9 to aim 10% lower). The inverse model predicts what x would achieve those lower objectives. This pushes the entire front toward better convergence — it's the most direct way to improve HV when the front shape is already good but solutions aren't fully converged.
     (c) **Extreme extension**: Extrapolate beyond the PF extremes to extend the front endpoints.
   - **Predict candidates**: Feed these target objectives through the inverse model to get predicted decision vectors. These are candidates that *should* achieve those objectives (or close to them).
   - **Why it works**: Instead of randomly searching decision space, you're asking "what x would give me this desired f?" The model learns the structure of the search space and can interpolate/extrapolate along the Pareto manifold. Works best for smooth, unimodal problems. For multimodal problems, combine with random restarts or use it on sub-regions.
   - **Budget split**: Divide your candidates across all three uses. For example: 40% front pushing, 30% gap filling, 30% extreme extension. Adjust based on what the front needs most (check gap sizes vs convergence potential).

4. **CMA-ES-inspired (Covariance Matrix Adaptation)**: Estimate the covariance matrix of PF solutions (or top-performing solutions). Sample new candidates from a multivariate Gaussian centered on the PF mean, with the estimated covariance. Adapt the step size based on improvement: shrink if improving (exploit), grow if stagnating (explore). Unlike DE, this captures the joint distribution of good solutions.

5. **Coordinate-wise Optimization**: For each PF solution, perturb one variable at a time (or one variable group at a time if non-separable) while keeping others fixed. Test multiple values for that variable. This is systematic and works well for separable or partially separable problems — it avoids the "curse of dimensionality" by optimizing one coordinate at a time.

Choose your strategy based on what you've learned about the problem:
- **Separable + unimodal** → coordinate-wise optimization, surrogate-inverse
- **Non-separable + unimodal** → PCA-EA, surrogate-inverse, CMA-ES
- **Separable + multimodal** → DE with large F, coordinate-wise with multiple restarts
- **Non-separable + multimodal** → PCA-EA with diversity, DE, mix of global + local
- **Stagnating** → switch to a completely different algorithm from this list
- Always combine your core algorithm with **gap filling** and **extreme pushing** sub-strategies.

## Stagnation Breaking (CRITICAL)
If HV has not improved (or improved by less than 0.1%) for 3+ consecutive generations, you are STUCK. This means your current assumptions about the problem may be WRONG. Do NOT keep refining the same approach harder — instead:

1. **Challenge your diagnostics**: Your separability/multimodality diagnosis might be incorrect. Re-examine with fresh eyes. Maybe you classified the problem as separable but it's actually coupled. Maybe you think it's unimodal but you're trapped in a local basin.
2. **Break your assumptions**: If you've been treating variables as independent, try coupling them. If you've been respecting variable groups, try ignoring them. If you've been exploiting near the PF, try random exploration far away from it.
3. **Increase exploration radius dramatically**: Double or triple your perturbation sizes. Dedicate the ENTIRE population to exploration (not just a fraction). Sample regions of decision space you've never visited.
4. **Try the algorithm you've used LEAST**: Look at your strategy history — which of the 5 search algorithms have you barely tried? Use that one. Your best algorithm for this problem might be one you haven't tested yet.
5. **Random restart with structure**: Generate half the population completely randomly (space-filling), and the other half by taking your best PF solutions and making LARGE perturbations (10-50% of the variable range, not 1-5%).

6. **Focus on Pareto IMPROVEMENT, not just expansion**: Stagnation often means you can't push the front further outward — but you CAN still improve HV by finding solutions that DOMINATE existing PF solutions (lower values on ALL objectives simultaneously). For each PF solution, try to find a nearby solution that beats it on every objective. Even small improvements (0.1% per objective) across many PF solutions add up to meaningful HV gains. This is convergence refinement — optimizing the convergence variables (the ones that don't control position along the front) toward their optimal values.

The goal when stuck is twofold: (1) discover NEW information about the problem by exploring, and (2) squeeze out convergence gains by improving existing PF solutions inward. Alternate between exploration and Pareto-improvement refinement."""


class MetisAgent:
    """Agentic black-box multi-objective optimizer.

    The LLM writes sampling code each generation based on evaluation history.

    Args:
        n_var: Number of decision variables.
        n_obj: Number of objectives.
        bounds: List of (lo, hi) per variable. Or single (lo, hi) for uniform bounds.
        population_size: Batch size per generation.
        model: Claude model alias/id to use (resolved by the `claude` CLI).
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        bounds,
        population_size: int = 120,
        model: str = None,
        initial_population: np.ndarray = None,
        max_elites: int = None,
        problem_context: str = None,
    ):
        self.n_var = n_var
        self.n_obj = n_obj
        self.pop_size = population_size
        self.model = model or _get_model()

        # Normalize bounds
        if isinstance(bounds, tuple) and len(bounds) == 2:
            self.bounds = [bounds] * n_var
        else:
            self.bounds = list(bounds)
        self.lb = self.bounds[0][0]
        self.ub = self.bounds[0][1]

        # State
        self.pf_X: List[np.ndarray] = []
        self.pf_F: List[np.ndarray] = []
        self._all_X: List[np.ndarray] = []
        self._all_F: List[np.ndarray] = []
        self.generation = 0
        self.hv_history: List[float] = []
        self.pf_history: List[List[List[float]]] = []  # PF objectives per generation
        self.ref: Optional[List[float]] = None
        self._token_usage = {"input": 0, "output": 0}
        self._last_code = ""
        self._initial_population = initial_population  # for fair comparison
        self.max_elites = max_elites  # None = unlimited
        self._problem_context = problem_context  # extra domain knowledge for LLM

        # Manual conversation transcript. The Claude Agent SDK is single-shot, so we fold
        # the running history into each prompt to preserve cross-generation memory.
        self._system = self._build_system()
        self._history: List[dict] = []  # [{"role": "user"/"assistant", "text": str}, ...]
        self._strategy_log: List[dict] = []  # {gen, code_summary, hv, hv_delta, error}
        self._max_history_turns = 20  # keep last N exchanges to bound the prompt size

    def _build_system(self) -> str:
        prompt = SYSTEM_PROMPT.format(
            n_var=self.n_var, n_obj=self.n_obj,
            lb=self.lb, ub=self.ub, pop_size=self.pop_size,
        )
        if self._problem_context:
            prompt += "\n\n## Problem Context (domain knowledge)\n" + self._problem_context
        return prompt

    def _render_history(self) -> str:
        """Fold the running transcript into a single prompt (the SDK is single-shot)."""
        blocks = []
        for turn in self._history:
            tag = "OPTIMIZATION REQUEST" if turn["role"] == "user" else "YOUR RESPONSE"
            blocks.append(f"===== {tag} =====\n{turn['text']}")
        blocks.append("Respond to the most recent OPTIMIZATION REQUEST above with your "
                      "LEARNINGS block and a single ```python code block ending in RESULT = X.")
        return "\n\n".join(blocks)

    def _build_context(self) -> str:
        g = self.generation
        n_total = len(self._all_F)

        if g == 0:
            return (f"Generation 0. No evaluations yet.\n"
                    f"Generate {self.pop_size} initial candidates to explore "
                    f"the {self.n_var}-dimensional space [{self.lb}, {self.ub}].")

        pf_F = np.array(self.pf_F)
        pf_X = np.array(self.pf_X)
        F_all = np.array(self._all_F)

        ctx = f"Generation {g}. Total evaluations: {n_total}.\n\n"

        # Pareto front summary
        n_pf = len(pf_F)
        ctx += f"## Pareto Front ({n_pf} solutions)\n"
        sort_idx = np.argsort(pf_F[:, 0])
        show = min(n_pf, 20)
        for rank, si in enumerate(sort_idx[:show]):
            obj_str = ", ".join(f"{v:.6f}" for v in pf_F[si])
            # Show a few key variables
            x = pf_X[si]
            x_str = ", ".join(f"{v:.4f}" for v in x[:6])
            ctx += f"  [{rank}] obj=({obj_str})  x[:6]=({x_str})\n"
        if n_pf > show:
            ctx += f"  ... +{n_pf - show} more solutions\n"

        # Objective statistics
        ctx += f"\n## Objective Ranges (across all {n_total} evaluations)\n"
        for j in range(self.n_obj):
            ctx += (f"  f{j+1}: min={F_all[:, j].min():.6f}, "
                    f"max={F_all[:, j].max():.6f}, "
                    f"median={np.median(F_all[:, j]):.6f}\n")

        # PF objective ranges
        ctx += f"\n## PF Objective Ranges\n"
        for j in range(self.n_obj):
            ctx += (f"  f{j+1}: [{pf_F[:, j].min():.6f}, {pf_F[:, j].max():.6f}]\n")

        # HV trend
        if len(self.hv_history) >= 2:
            recent = self.hv_history[-5:]
            deltas = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            ctx += f"\n## HV Trend (last 5): {[f'{h:.4f}' for h in recent]}\n"
            ctx += f"  Deltas: {[f'{d:+.4f}' for d in deltas]}\n"
        elif self.hv_history:
            ctx += f"\n## HV: {self.hv_history[-1]:.4f}\n"

        # PF uniformity / gap analysis
        if n_pf >= 3:
            sort_idx_pf = np.argsort(pf_F[:, 0])
            sorted_F = pf_F[sort_idx_pf]
            # Compute gaps between consecutive PF solutions
            gaps = np.linalg.norm(np.diff(sorted_F, axis=0), axis=1)
            top_gaps = np.argsort(gaps)[-5:][::-1]
            ctx += f"\n## PF Gaps (largest spacing between consecutive solutions, sorted by f1)\n"
            for gi in top_gaps:
                f_left = ", ".join(f"{v:.6f}" for v in sorted_F[gi])
                f_right = ", ".join(f"{v:.6f}" for v in sorted_F[gi + 1])
                ctx += f"  Gap {gi}-{gi+1}: distance={gaps[gi]:.6f}  between ({f_left}) and ({f_right})\n"

        # Variable analysis
        if n_pf >= 3:
            var_std = np.std(pf_X, axis=0)
            top_k = min(15, self.n_var)
            top_vars = np.argsort(var_std)[-top_k:][::-1]
            ctx += f"\n## Most Influential Variables (highest variance across PF)\n"
            ctx += f"  Indices: {top_vars.tolist()}\n"
            ctx += f"  Std devs: {var_std[top_vars].round(4).tolist()}\n"

            # Variable means on PF
            pf_mean = np.mean(pf_X, axis=0)
            ctx += f"  PF mean values: {pf_mean[top_vars].round(4).tolist()}\n"

        # Correlation between variables and objectives (for top vars)
        if n_total >= 20 and n_pf >= 3:
            X_all = np.array(self._all_X)
            ctx += f"\n## Variable-Objective Correlations (top vars)\n"
            top5 = np.argsort(np.std(pf_X, axis=0))[-5:][::-1]
            for vi in top5:
                corrs = []
                for j in range(self.n_obj):
                    c = np.corrcoef(X_all[:, vi], F_all[:, j])[0, 1]
                    corrs.append(f"f{j+1}:{c:+.3f}")
                ctx += f"  x[{vi}]: {', '.join(corrs)}\n"

        ctx += f"\nGenerate {self.pop_size} candidates for generation {g}."
        ctx += f" Remember to run your separability/multimodality diagnostics in your code."
        return ctx

    def _extract_learnings(self, text: str) -> str:
        """Extract LEARNINGS section from response."""
        if "LEARNINGS:" in text:
            start = text.find("LEARNINGS:") + len("LEARNINGS:")
            # End at code block or end of text
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
            return text[start:].strip()
        return ""

    def _extract_code(self, text: str) -> str:
        """Extract Python code from response."""
        # Look for ```python blocks (case-insensitive, with optional whitespace)
        import re
        m = re.search(r"```[Pp]ython\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Any fenced code block
        m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: lines that look like Python code (have RESULT =)
        lines = []
        capture = False
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "# Strategy", "RESULT", "X ", "X=")) or capture:
                lines.append(line)
                capture = True
        if lines:
            return "\n".join(lines).strip()
        print(f"  [WARN] Could not extract code from response:\n{text[:500]}", flush=True)
        return text.strip()

    def _execute_code(self, code: str, seed: int) -> np.ndarray:
        """Execute sampling code in a sandboxed namespace."""
        # If code appears truncated (no RESULT assignment), try to fix it
        if "RESULT" not in code:
            # Find the last assigned array variable and add RESULT = it
            import re
            # Look for patterns like "X = ...", "candidates = ...", etc.
            last_var = None
            for m in re.finditer(r'^(\w+)\s*=\s*', code, re.MULTILINE):
                candidate = m.group(1)
                if candidate not in ('np', 'rng', 'n_var', 'n_obj', 'pop_size',
                                     'bounds', 'pf_X', 'pf_F', 'all_X', 'all_F',
                                     'generation', 'True', 'False', 'None'):
                    last_var = candidate
            if last_var:
                code += f"\nRESULT = {last_var}"
                print(f"  [WARN] Code truncated (no RESULT). Appended RESULT = {last_var}", flush=True)

        rng = np.random.RandomState(seed)
        namespace = {
            "np": np,
            "rng": rng,
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "pop_size": self.pop_size,
            "bounds": self.bounds,
            "pf_X": np.array(self.pf_X) if self.pf_X else np.empty((0, self.n_var)),
            "pf_F": np.array(self.pf_F) if self.pf_F else np.empty((0, self.n_obj)),
            "all_X": np.array(self._all_X) if self._all_X else np.empty((0, self.n_var)),
            "all_F": np.array(self._all_F) if self._all_F else np.empty((0, self.n_obj)),
            "generation": self.generation,
        }

        exec(code, namespace)

        result = namespace.get("RESULT")
        if result is None:
            # LLM sometimes forgets RESULT = X; try common variable names
            for name in ("X", "candidates", "result", "X_new", "samples"):
                val = namespace.get(name)
                if val is not None and isinstance(val, np.ndarray) and val.ndim == 2:
                    if val.shape == (self.pop_size, self.n_var):
                        print(f"  [WARN] RESULT not set, using '{name}' instead", flush=True)
                        result = val
                        break
        if result is None:
            # Last resort: scan namespace for any array with the right shape
            for name, val in namespace.items():
                if name.startswith('_') or name in ('np', 'rng', 'bounds',
                    'pf_X', 'pf_F', 'all_X', 'all_F'):
                    continue
                if isinstance(val, np.ndarray) and val.shape == (self.pop_size, self.n_var):
                    print(f"  [WARN] RESULT not set, using '{name}' (shape match) as fallback", flush=True)
                    result = val
                    break
        if result is None:
            raise ValueError("Code did not set RESULT variable")

        result = np.asarray(result, dtype=float)
        if result.shape != (self.pop_size, self.n_var):
            # Try to fix shape
            if result.size == self.pop_size * self.n_var:
                result = result.reshape(self.pop_size, self.n_var)
            else:
                raise ValueError(f"RESULT shape {result.shape} != ({self.pop_size}, {self.n_var})")

        # Clip to bounds
        result = np.clip(result, self.lb, self.ub)
        return result

    def _build_feedback(self) -> str:
        """Build feedback message about the last generation's results."""
        if not self._strategy_log:
            return ""

        last = self._strategy_log[-1]
        fb = f"\n## Results from Generation {last['gen']}\n"
        fb += f"- HV: {last['hv']:.4f}"
        if last['hv_delta'] is not None:
            sign = "+" if last['hv_delta'] >= 0 else ""
            fb += f" ({sign}{last['hv_delta']:.4f} change)"
            if last['hv_delta'] > 0:
                fb += " ✅ IMPROVED"
            elif last['hv_delta'] == 0:
                fb += " ⚠️ NO CHANGE"
            else:
                fb += " ❌ REGRESSED"
        fb += "\n"
        if last.get('error'):
            fb += f"- ⚠️ Your code had an error: {last['error']}\n"
            fb += "- Fallback was used (PF perturbation). Fix your code this time.\n"
        if last.get('new_pf_count') is not None:
            fb += f"- New PF solutions found this gen: {last['new_pf_count']}\n"
        if last.get('code_summary'):
            fb += f"- Your strategy was: {last['code_summary']}\n"

        # Show recent strategy performance
        if len(self._strategy_log) >= 3:
            fb += "\n## Strategy Performance History\n"
            for entry in self._strategy_log[-10:]:
                delta_str = f"{entry['hv_delta']:+.4f}" if entry['hv_delta'] is not None else "N/A"
                err_str = " [ERROR]" if entry.get('error') else ""
                fb += f"  Gen {entry['gen']}: HV={entry['hv']:.4f} (Δ={delta_str}){err_str} — {entry.get('code_summary', '?')}\n"

        return fb

    def ask(self) -> np.ndarray:
        """Ask the agent to generate candidates (Claude, with folded conversation memory)."""
        # If an initial population was provided, use it for gen 0 (fair comparison) and
        # inject a synthetic gen-0 exchange so the agent has context when gen 1 starts.
        if self.generation == 0 and self._initial_population is not None:
            gen0_user = (f"Generation 0. No evaluations yet.\n"
                         f"Generate {self.pop_size} initial candidates to explore "
                         f"the {self.n_var}-dimensional space [{self.lb}, {self.ub}].")
            gen0_model = ("```python\n# Strategy: Uniform random initialization\n"
                          f"X = rng.uniform({self.lb}, {self.ub}, (pop_size, n_var))\n"
                          "RESULT = X\n```")
            self._history.append({"role": "user", "text": gen0_user})
            self._history.append({"role": "assistant", "text": gen0_model})
            self._pending_summary = "Shared initial population (injected)"
            self._pending_error = None
            self._pending_learnings = ""
            return np.asarray(self._initial_population)

        context = self._build_context()

        # Add feedback from last generation
        feedback = self._build_feedback()
        if feedback:
            context = feedback + "\n" + context

        self._history.append({"role": "user", "text": context})
        # Trim to bound prompt size (drop oldest full exchanges)
        while len(self._history) > self._max_history_turns * 2:
            self._history.pop(0)

        text = ""
        try:
            text, thinking = claude_llm.invoke(self._system, self._render_history(),
                                               model=self.model)
            if thinking:
                print(f"\n  \033[32m[THINKING] {thinking.strip()}\033[0m", flush=True)
            self._history.append({"role": "assistant", "text": text})

            learnings = self._extract_learnings(text)
            code = self._extract_code(text)
            self._last_code = code
            self._last_learnings = learnings

            seed = self.generation * 1000 + 42
            result = self._execute_code(code, seed)

            # Extract strategy summary (first comment line)
            summary = ""
            for line in code.split("\n"):
                line = line.strip()
                if line.startswith("#") and "Strategy" in line:
                    summary = line.lstrip("# ").strip()
                    break
            if not summary:
                for line in code.split("\n"):
                    if line.strip().startswith("#"):
                        summary = line.strip().lstrip("# ")[:80]
                        break

            self._pending_summary = summary
            self._pending_error = None
            self._pending_learnings = learnings
            return result

        except Exception as e:
            error_msg = str(e)
            print(f"  Agent error: {error_msg}", flush=True)
            if text and (not self._history or self._history[-1]["role"] != "assistant"):
                self._history.append({"role": "assistant", "text": text})

            self._pending_summary = "FALLBACK (code error)"
            self._pending_error = error_msg

            # Fallback: random or perturb PF
            if self.pf_X:
                X = np.array(self.pf_X)
                idx = np.random.randint(len(X), size=self.pop_size)
                noise = np.random.randn(self.pop_size, self.n_var) * 0.05
                return np.clip(X[idx] + noise, self.lb, self.ub)
            return np.random.uniform(self.lb, self.ub, (self.pop_size, self.n_var))

    def tell(self, X: np.ndarray, F: np.ndarray):
        """Report evaluation results."""
        old_pf_size = len(self.pf_X)

        for i in range(len(X)):
            self._all_X.append(X[i].copy())
            self._all_F.append(F[i].copy())

        # Update Pareto front
        F_all = np.array(self._all_F)
        X_all = np.array(self._all_X)
        pf_idx = _pareto_front_indices(F_all)
        self.pf_X = [X_all[i].copy() for i in pf_idx]
        self.pf_F = [F_all[i].copy() for i in pf_idx]

        # Prune to max_elites using crowding distance (uniform spread)
        if self.max_elites is not None and len(self.pf_X) > self.max_elites:
            self.pf_X, self.pf_F = _prune_to_size(self.pf_X, self.pf_F, self.max_elites)

        new_pf_count = max(0, len(self.pf_X) - old_pf_size)

        # Snapshot elite set for this generation
        self.pf_history.append([f.tolist() for f in self.pf_F])

        # Compute HV
        if self.ref is not None:
            try:
                from agentic_optimizer.hypervolume import hypervolume as hv_calc
                from agentic_optimizer.individual import Individual
                elites = []
                for i in range(len(self.pf_X)):
                    ind = Individual()
                    ind.dofs = self.pf_X[i]
                    ind.objs = self.pf_F[i]
                    elites.append(ind)
                hv = hv_calc(elites, self.ref)
                self.hv_history.append(hv)
            except Exception:
                pass

        # Log strategy results
        hv = self.hv_history[-1] if self.hv_history else 0
        hv_delta = None
        if len(self.hv_history) >= 2:
            hv_delta = self.hv_history[-1] - self.hv_history[-2]

        self._strategy_log.append({
            "gen": self.generation,
            "hv": hv,
            "hv_delta": hv_delta,
            "new_pf_count": new_pf_count,
            "code_summary": getattr(self, '_pending_summary', ''),
            "error": getattr(self, '_pending_error', None),
            "learnings": getattr(self, '_pending_learnings', ''),
        })

        self.generation += 1

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current Pareto front."""
        return np.array(self.pf_X), np.array(self.pf_F)

    @property
    def token_usage(self):
        return self._token_usage

    @property
    def last_code(self):
        return self._last_code
