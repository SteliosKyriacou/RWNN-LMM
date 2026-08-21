# Strategy: Coordinate-wise group refinement from feasible PF baselines (conservative)
import numpy as np

if len(all_X) > 5:
    ac = np.corrcoef(all_X.T)
    ac_mean = np.nanmean(np.abs(ac[np.triu_indices_from(ac,1)]))
    if len(pf_X) > 3:
        pc = np.corrcoef(pf_X.T)
        pc_mean = np.nanmean(np.abs(pc[np.triu_indices_from(pc,1)]))
        print(f"Excess corr PF-all: {pc_mean-ac_mean:.4f} (small PF sample, low confidence)")

def clip(x): return np.clip(x, 0.0, 1.0)

cands = []
base = pf_X if len(pf_X) > 0 else all_X[np.argsort(all_F[:,0])[:4]]
nb = len(base)
order = np.argsort(pf_F[:,0]) if len(pf_F) > 1 else None

for i in range(nb):
    b = base[i]

    # 1) small local Gaussian perturbation (fine local search, low risk)
    x1 = b + rng.normal(0, 0.03, n_var)
    cands.append(clip(x1))

    # 2) MoE push: pick 3-5 random FFN slots, push toward high experts + low topk
    x2 = b.copy()
    slots = rng.choice(16, rng.randint(3,6), replace=False)
    for s in slots:
        x2[65+s] = np.clip(x2[65+s] + rng.uniform(0.15, 0.3), 0, 1)  # more experts
        x2[81+s] = np.clip(x2[81+s] - rng.uniform(0.1, 0.25), 0, 1)  # lower topk
    x2 += rng.normal(0, 0.01, n_var)
    cands.append(clip(x2))

    # 3) Skip push: increase long-range residual skips slightly (cheap gain)
    x3 = b.copy()
    x3[49:65] = np.clip(x3[49:65] + rng.uniform(0.05, 0.15, 16), 0, 1)
    x3 += rng.normal(0, 0.015, n_var)
    cands.append(clip(x3))

    # 4) Gentle depth/attn-ffn ordering nudge (small magnitude, avoid extreme collapse)
    x4 = b.copy()
    x4[0] = np.clip(x4[0] + rng.uniform(-0.08, 0.08), 0, 1)
    x4[1:17] = np.clip(x4[1:17] + rng.normal(0, 0.05, 16), 0, 1)
    x4[17:33] = np.clip(x4[17:33] + rng.normal(0, 0.05, 16), 0, 1)
    cands.append(clip(x4))

    # 5) Gap-fill interpolation toward neighbor on PF (small step), else extra local variant
    if order is not None and nb > 1:
        j = order[(np.where(order==i)[0][0]+1) % nb] if i in order else (i+1)%nb
        t = rng.uniform(0.2, 0.4)
        x5 = b*(1-t) + base[j]*t + rng.normal(0, 0.02, n_var)
    else:
        x5 = b + rng.normal(0, 0.05, n_var)
    cands.append(clip(x5))

while len(cands) < pop_size:
    cands.append(clip(base[rng.randint(nb)] + rng.normal(0, 0.04, n_var)))

X = np.array(cands[:pop_size])
RESULT = X