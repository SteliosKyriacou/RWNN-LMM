# Strategy: Stagnation-break: diversity restore (random) + light local refine + gap-fill
import numpy as np

def clip(x): return np.clip(x, 0.0, 1.0)

if len(all_X) > 5 and len(pf_X) > 3:
    ac = np.corrcoef(all_X.T); ac_mean = np.nanmean(np.abs(ac[np.triu_indices_from(ac,1)]))
    pc = np.corrcoef(pf_X.T); pc_mean = np.nanmean(np.abs(pc[np.triu_indices_from(pc,1)]))
    print(f"Excess corr PF-all: {pc_mean-ac_mean:.4f}")
    Fn = (pf_F - pf_F.min(0)) / (np.ptp(pf_F,axis=0)+1e-9)
    Xn = (pf_X - all_X.min(0)) / (np.ptp(all_X,axis=0)+1e-9)
    fd = np.linalg.norm(Fn[:,None]-Fn[None,:], axis=-1)
    xd = np.linalg.norm(Xn[:,None]-Xn[None,:], axis=-1)
    iu = np.triu_indices(len(pf_X),1)
    if len(iu[0])>0:
        thresh = np.percentile(fd[iu], 25)
        cm = fd[iu] < thresh
        if cm.sum()>0:
            ratio = xd[iu][cm].mean() / (xd[iu].mean()+1e-9)
            print(f"Multimodality ratio: {ratio:.3f}")
if len(pf_X)>0:
    print(f"Expert-var mean std PF: {pf_X[:,65:81].std(0).mean():.4f}")
print("HV declining 4/5 gens -> STAGNATION BREAK: restoring diversity via random-heavy gen")

cands = []
base = pf_X if len(pf_X)>=3 else all_X[np.argsort(all_F[:,0])[:6]]
nb = len(base)

# 1) Pure random space-filling (10) -- restore diversity, gen0 style but with domain-informed ranges
for _ in range(10):
    x = rng.uniform(0,1,n_var)
    # keep MoE loosely diverse but not forced to extremes
    x[65:81] = rng.choice([0.15,0.4,0.6,0.9], 16) + rng.normal(0,0.05,16)
    x[81:97] = rng.choice([0.2,0.7], 16) + rng.normal(0,0.05,16)
    cands.append(clip(x))

# 2) Small local refinement around PF solutions (5) -- tiny steps only (worked best so far)
for _ in range(5):
    b = base[rng.randint(nb)]
    x = b + rng.normal(0, 0.025, n_var)
    cands.append(clip(x))

# 3) Gap-fill at persistent largest gap (3) -- wider t range, bigger noise to escape stuck region
if len(pf_F) >= 2:
    order = np.argsort(pf_F[:,0])
    dists = [np.linalg.norm(pf_F[order[i+1]]-pf_F[order[i]]) for i in range(len(order)-1)]
    if dists:
        gi = np.argmax(dists)
        a = pf_X[order[gi]]; b = pf_X[order[gi+1]]
        for _ in range(3):
            t = rng.uniform(0.1, 0.9)
            child = a*(1-t) + b*t + rng.normal(0, 0.05, n_var)
            cands.append(clip(child))

# 4) Mild extreme pushes -- single-slot MoE nudges (not all 16 slots at once)
bestf2 = pf_X[np.argmin(pf_F[:,1])] if len(pf_F)>0 else base[-1]
for _ in range(2):
    x = bestf2.copy()
    s = rng.choice(16, 3, replace=False)
    for si in s:
        x[65+si] = np.clip(x[65+si] + rng.uniform(0.05,0.15), 0, 1)
        x[81+si] = np.clip(x[81+si] - rng.uniform(0.05,0.15), 0, 1)
    x += rng.normal(0,0.015,n_var)
    cands.append(clip(x))

while len(cands) < pop_size:
    cands.append(clip(rng.uniform(0,1,n_var)))

X = np.array(cands[:pop_size])
RESULT = X