# Strategy: Coordinate-wise group optimization + gap-fill + light random diversity
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
    print(f"Expert-var mean std PF: {pf_X[:,65:81].std(0).mean():.4f} (saturation check)")

cands = []
base = pf_X if len(pf_X)>=3 else all_X[np.argsort(all_F[:,0])[:6]]
nb = len(base)

groups = {
    'depth': [0],
    'attn': list(range(1,17)),
    'ffn': list(range(17,33)),
    'act': list(range(33,49)),
    'skip': list(range(49,65)),
    'experts': list(range(65,81)),
    'topk': list(range(81,97)),
}
gkeys = list(groups.keys())

# 1) Coordinate-wise: for each PF sol, perturb ONE group at a time (2 per sol = 8)
for i in range(nb):
    b = base[i]
    chosen = rng.choice(len(gkeys), min(2,len(gkeys)), replace=False)
    for gk_idx in chosen:
        gk = gkeys[gk_idx]
        idxs = groups[gk]
        x = b.copy()
        step = rng.uniform(0.1, 0.3)
        direction = rng.choice([-1,1])
        x[idxs] = np.clip(x[idxs] + direction*step + rng.normal(0,0.03,len(idxs)), 0, 1)
        cands.append(clip(x))

# 2) Gap-fill largest gap (1-2), heavy focus (5)
if len(pf_F) >= 2:
    order = np.argsort(pf_F[:,0])
    dists = [np.linalg.norm(pf_F[order[k+1]]-pf_F[order[k]]) for k in range(len(order)-1)]
    if dists:
        gi = np.argmax(dists)
        a = pf_X[order[gi]]; b2 = pf_X[order[gi+1]]
        for _ in range(5):
            t = rng.uniform(0.1, 0.9)
            child = a*(1-t) + b2*t + rng.normal(0, 0.04, n_var)
            cands.append(clip(child))

# 3) light random diversity for exploration (4)
for _ in range(4):
    x = rng.uniform(0,1,n_var)
    x[65:81] = rng.choice([0.15,0.4,0.6,0.9], 16) + rng.normal(0,0.06,16)
    x[81:97] = rng.choice([0.2,0.7], 16) + rng.normal(0,0.06,16)
    cands.append(clip(x))

# 4) tiny local refine on best f1 and best f2 (2)
if len(pf_F)>0:
    bf1 = pf_X[np.argmin(pf_F[:,0])]
    bf2 = pf_X[np.argmin(pf_F[:,1])]
    cands.append(clip(bf1 + rng.normal(0,0.02,n_var)))
    cands.append(clip(bf2 + rng.normal(0,0.02,n_var)))

while len(cands) < pop_size:
    cands.append(clip(base[rng.randint(nb)] + rng.normal(0,0.05,n_var)))

X = np.array(cands[:pop_size])
RESULT = X