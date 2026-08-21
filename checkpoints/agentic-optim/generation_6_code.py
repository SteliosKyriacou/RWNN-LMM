# Strategy: CMA-ES-inspired sampling + gap-fill + mild extreme pushing
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

cands = []
base = pf_X if len(pf_X)>=3 else all_X[np.argsort(all_F[:,0])[:6]]
nb = len(base)

# CMA-ES-inspired: mean+cov of PF, sample with moderate step (HV declining -> keep exploring, don't shrink too much)
mean = base.mean(0)
cov = np.cov(base.T) + np.eye(n_var)*1e-4
step = 0.8  # exploration scale since recent regressions
try:
    L = np.linalg.cholesky(cov*step*step*0.02 + np.eye(n_var)*1e-6)
except np.linalg.LinAlgError:
    L = np.eye(n_var)*0.05

for _ in range(8):
    z = rng.normal(0,1,n_var)
    child = mean + L@z
    cands.append(clip(child))

# Gap-fill at largest gap (2-3) between f2=131M and f2=50M
if len(pf_F)>=2:
    order = np.argsort(pf_F[:,0])
    dists = [np.linalg.norm(pf_F[order[i+1]]-pf_F[order[i]]) for i in range(len(order)-1)]
    if dists:
        gi = np.argmax(dists)
        a = pf_X[order[gi]]; b = pf_X[order[gi+1]]
        for _ in range(4):
            t = rng.uniform(0.15,0.85)
            child = a*(1-t)+b*t + rng.normal(0,0.02,n_var)
            cands.append(clip(child))
        # second largest gap
        d2 = list(dists)
        d2[gi] = -1
        gi2 = np.argmax(d2)
        if d2[gi2] > 0:
            a2 = pf_X[order[gi2]]; b2 = pf_X[order[gi2+1]]
            for _ in range(3):
                t = rng.uniform(0.2,0.8)
                child = a2*(1-t)+b2*t + rng.normal(0,0.02,n_var)
                cands.append(clip(child))

# Mild extreme push: lowest-f1 solution -> nudge toward lower loss (small step)
bestf1 = pf_X[np.argmin(pf_F[:,0])] if len(pf_F)>0 else base[0]
for _ in range(3):
    x = bestf1.copy()
    x[0] = np.clip(x[0] + rng.uniform(0.0,0.1), 0, 1)  # slightly deeper
    x[49:65] = np.clip(x[49:65] + rng.uniform(0.0,0.1,16), 0, 1)  # more skips
    x += rng.normal(0,0.02,n_var)
    cands.append(clip(x))

# Mild extreme push: lowest-f2 solution -> nudge toward lower FLOPs (small step, no aggressive MoE)
bestf2 = pf_X[np.argmin(pf_F[:,1])] if len(pf_F)>0 else base[-1]
for _ in range(2):
    x = bestf2.copy()
    x[1:33] = np.clip(x[1:33] - rng.uniform(0.0,0.08,32), 0, 1)  # slightly more bypass
    x += rng.normal(0,0.02,n_var)
    cands.append(clip(x))

while len(cands) < pop_size:
    cands.append(clip(base[rng.randint(nb)] + rng.normal(0,0.05,n_var)))

X = np.array(cands[:pop_size])
RESULT = X