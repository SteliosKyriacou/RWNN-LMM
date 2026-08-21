# Strategy: Diversity-heavy random + DE (light) + gap-fill focus on gap1-2 + tiny local refine
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
    for gi in [65,70,74,78]:
        print(f"var{gi} PF vals: {pf_X[:,gi]}")

cands = []
base = pf_X if len(pf_X)>=3 else all_X[np.argsort(all_F[:,0])[:6]]
nb = len(base)

# 1) Diversity random space-filling (8) - domain-informed but broad
for _ in range(8):
    x = rng.uniform(0,1,n_var)
    x[65:81] = rng.choice([0.15,0.4,0.6,0.9], 16) + rng.normal(0,0.06,16)
    x[81:97] = rng.choice([0.2,0.7], 16) + rng.normal(0,0.06,16)
    cands.append(clip(x))

# 2) DE from PF (light F, no forced extremes) (6)
if nb >= 3:
    for _ in range(6):
        idx = rng.choice(nb, 3, replace=nb<3)
        a,b,c = base[idx[0]], base[idx[1]], base[idx[2]]
        F = rng.uniform(0.3, 0.7)
        mutant = a + F*(b-c)
        parent = base[rng.randint(nb)]
        mask = rng.rand(n_var) < 0.5
        child = np.where(mask, mutant, parent)
        cands.append(clip(child))

# 3) Gap-fill largest persistent gap (1-2) heavily (4)
if len(pf_F) >= 2:
    order = np.argsort(pf_F[:,0])
    dists = [np.linalg.norm(pf_F[order[i+1]]-pf_F[order[i]]) for i in range(len(order)-1)]
    if dists:
        gi = np.argmax(dists)
        a = pf_X[order[gi]]; b = pf_X[order[gi+1]]
        for _ in range(4):
            t = rng.uniform(0.1, 0.9)
            child = a*(1-t) + b*t + rng.normal(0, 0.04, n_var)
            cands.append(clip(child))

# 4) tiny local refine (2)
for _ in range(2):
    b = base[rng.randint(nb)]
    cands.append(clip(b + rng.normal(0, 0.02, n_var)))

while len(cands) < pop_size:
    x = rng.uniform(0,1,n_var)
    cands.append(clip(x))

X = np.array(cands[:pop_size])
RESULT = X