# Strategy: Diversity-heavy random + light DE + gap-fill (gen7 winning formula, revived) + extreme pushes
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

# 1) Diversity random space-filling (8)
for _ in range(8):
    x = rng.uniform(0,1,n_var)
    x[65:81] = rng.choice([0.15,0.4,0.6,0.9], 16) + rng.normal(0,0.06,16)
    x[81:97] = rng.choice([0.2,0.7], 16) + rng.normal(0,0.06,16)
    cands.append(clip(x))

# 2) DE from PF (light F) (6)
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

# 3) Gap-fill largest gap (0-1, 86.5M) heavily (3)
if len(pf_F) >= 2:
    order = np.argsort(pf_F[:,0])
    dists = [np.linalg.norm(pf_F[order[k+1]]-pf_F[order[k]]) for k in range(len(order)-1)]
    if dists:
        gi = np.argmax(dists)
        a = pf_X[order[gi]]; b2 = pf_X[order[gi+1]]
        for _ in range(3):
            t = rng.uniform(0.1, 0.9)
            child = a*(1-t) + b2*t + rng.normal(0, 0.04, n_var)
            cands.append(clip(child))

# 4) Extreme push f1 (lower loss): nudge lowest-f1 sol deeper/more skip (1)
if len(pf_F)>0:
    bf1 = pf_X[np.argmin(pf_F[:,0])].copy()
    bf1[0] = np.clip(bf1[0] + rng.uniform(0.0,0.15), 0, 1)
    bf1[49:65] = np.clip(bf1[49:65] + rng.uniform(0.0,0.1,16), 0, 1)
    bf1 += rng.normal(0,0.02,n_var)
    cands.append(clip(bf1))

# 5) Extreme push f2 (lower FLOPs): nudge lowest-f2 sol toward more bypass, single-slot MoE tweaks (1)
if len(pf_F)>0:
    bf2 = pf_X[np.argmin(pf_F[:,1])].copy()
    bf2[1:33] = np.clip(bf2[1:33] - rng.uniform(0.0,0.08,32), 0, 1)
    s = rng.choice(16, 3, replace=False)
    for si in s:
        bf2[65+si] = np.clip(bf2[65+si] + rng.uniform(0.05,0.15), 0, 1)
        bf2[81+si] = np.clip(bf2[81+si] - rng.uniform(0.05,0.15), 0, 1)
    bf2 += rng.normal(0,0.015,n_var)
    cands.append(clip(bf2))

# 6) tiny local refine (1)
b = base[rng.randint(nb)]
cands.append(clip(b + rng.normal(0, 0.02, n_var)))

while len(cands) < pop_size:
    x = rng.uniform(0,1,n_var)
    cands.append(clip(x))

X = np.array(cands[:pop_size])
RESULT = X