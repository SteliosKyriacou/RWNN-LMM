# Strategy: DE-from-PF + domain-guided extremes/gap-fill (MoE sweet spot, front-loaded attention)
import numpy as np

if len(all_X) > 3:
    ac = np.corrcoef(all_X.T)
    ac_mean = np.nanmean(np.abs(ac[np.triu_indices_from(ac,1)]))
    print(f"AllX mean|corr|: {ac_mean:.4f} (baseline)")
if len(pf_X) > 3:
    pc = np.corrcoef(pf_X.T)
    pc_mean = np.nanmean(np.abs(pc[np.triu_indices_from(pc,1)]))
    print(f"PF mean|corr|: {pc_mean:.4f} vs all -> excess={pc_mean-ac_mean:.4f}")

cands = []

def clip(x): return np.clip(x, 0.0, 1.0)

# 1) DE mutants from PF (4)
if len(pf_X) >= 3:
    for _ in range(4):
        idx = rng.choice(len(pf_X), 3, replace=True)
        a, b, c = pf_X[idx[0]], pf_X[idx[1]], pf_X[idx[2]]
        F = rng.uniform(0.4, 0.9)
        mutant = a + F * (b - c)
        parent = pf_X[rng.randint(len(pf_X))]
        mask = rng.rand(n_var) < 0.5
        child = np.where(mask, mutant, parent)
        cands.append(clip(child))
else:
    for _ in range(4):
        cands.append(rng.uniform(0,1,n_var))

# 2) Extreme push f2 (active-FLOPs): MoE sweet spot high experts/low topk, more bypass, shallow-ish (4)
for _ in range(4):
    x = rng.uniform(0,1,n_var)
    x[0] = rng.uniform(0.1, 0.4)  # shallower depth
    x[1:17] = rng.uniform(0.55, 0.95, 16)  # attention mostly active early anyway
    for i in range(1,17):
        if i > 10: x[i-1+1] = rng.uniform(0.0, 0.4)  # bypass late attention
    x[17:33] = rng.uniform(0.2, 0.6, 16)  # some FFN bypass
    x[49:65] = rng.uniform(0.5, 1.0, 16)  # skips cheap, keep on
    x[65:81] = rng.uniform(0.78, 1.0, 16)  # n_experts=8
    x[81:97] = rng.uniform(0.0, 0.35, 16)  # top_k=1
    x[33:49] = rng.uniform(0,1,16)  # mixed activations
    cands.append(clip(x))

# 3) Extreme push f1 (loss): deeper, dense-ish, front-loaded attention early/FFN later (4)
for _ in range(4):
    x = rng.uniform(0,1,n_var)
    x[0] = rng.uniform(0.6, 1.0)  # deeper
    x[1:17] = np.linspace(0.95, 0.4, 16) + rng.normal(0,0.05,16)  # attn front-loaded
    x[17:33] = np.linspace(0.3, 0.95, 16) + rng.normal(0,0.05,16)  # ffn back-loaded
    x[49:65] = rng.uniform(0.6, 1.0, 16)  # skips on
    x[65:81] = rng.uniform(0.6, 1.0, 16)  # high experts (capacity, cheap on flops)
    x[81:97] = rng.uniform(0.0, 0.6, 16)  # mixed topk
    x[33:49] = rng.uniform(0,1,16)
    cands.append(clip(x))

# 4) Gap-fill / interpolation between PF neighbors sorted by f1 (4)
if len(pf_X) >= 2:
    order = np.argsort(pf_F[:,0])
    for i in range(min(4, len(order)-1)):
        a = pf_X[order[i]]; b = pf_X[order[i+1]]
        t = rng.uniform(0.3, 0.7)
        child = a*(1-t) + b*t
        child += rng.normal(0, 0.03, n_var)
        cands.append(clip(child))
    while len(cands) < 16:
        cands.append(rng.uniform(0,1,n_var))
else:
    while len(cands) < 16:
        cands.append(rng.uniform(0,1,n_var))

# 5) Fill remaining with diverse random MoE-forced exploration
while len(cands) < pop_size:
    x = rng.uniform(0,1,n_var)
    # force MoE diversity: random combos of experts in {2,4,8}, topk in {1,2}
    exp_choice = rng.choice([0.4, 0.6, 0.9], 16)
    topk_choice = rng.choice([0.2, 0.7], 16)
    x[65:81] = exp_choice + rng.normal(0,0.03,16)
    x[81:97] = topk_choice + rng.normal(0,0.03,16)
    cands.append(clip(x))

X = np.array(cands[:pop_size])
RESULT = X