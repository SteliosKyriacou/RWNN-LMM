# Strategy: PCA-EA (crossover+mutation in PF PCA space) + MoE saturation breaker + gap-fill
import numpy as np

def clip(x): return np.clip(x, 0.0, 1.0)

# Diagnostics
if len(all_X) > 5 and len(pf_X) > 3:
    ac = np.corrcoef(all_X.T); ac_mean = np.nanmean(np.abs(ac[np.triu_indices_from(ac,1)]))
    pc = np.corrcoef(pf_X.T); pc_mean = np.nanmean(np.abs(pc[np.triu_indices_from(pc,1)]))
    print(f"Excess corr PF-all: {pc_mean-ac_mean:.4f}")
    # multimodality check
    Fn = (pf_F - pf_F.min(0)) / (pf_F.ptp(0)+1e-9)
    Xn = (pf_X - all_X.min(0)) / (all_X.ptp(0)+1e-9)
    fd = np.linalg.norm(Fn[:,None]-Fn[None,:], axis=-1)
    xd = np.linalg.norm(Xn[:,None]-Xn[None,:], axis=-1)
    iu = np.triu_indices(len(pf_X),1)
    if len(iu[0])>0:
        thresh = np.percentile(fd[iu], 25)
        close_mask = fd[iu] < thresh
        if close_mask.sum()>0:
            ratio = xd[iu][close_mask].mean() / (xd[iu].mean()+1e-9)
            print(f"Multimodality ratio: {ratio:.3f}")
# saturation check
if len(pf_X)>0:
    exp_std = pf_X[:,65:81].std(0).mean()
    print(f"Expert-var mean std across PF: {exp_std:.4f} (low=saturated)")

cands = []
base = pf_X if len(pf_X)>=3 else all_X[np.argsort(all_F[:,0])[:6]]
nb = len(base)

# PCA on PF
mean = base.mean(0)
centered = base - mean
U,S,Vt = np.linalg.svd(centered, full_matrices=False)
k = min(5, len(S))
P = Vt[:k].T  # n_var x k
var_w = S[:k] / (S[:k].sum()+1e-9)

# 1) PCA mutation (6): larger noise top comps, smaller bottom
for _ in range(6):
    b = base[rng.randint(nb)]
    coef = (b-mean) @ P
    noise = rng.normal(0,1,k) * (0.15*var_w[0]/ (var_w+1e-6)).clip(0.05,0.5) * S.mean()*0.02
    coef2 = coef + rng.normal(0,1,k)*np.linspace(0.35,0.05,k)*np.abs(S[:k]).mean()*0.05
    child = mean + P@coef2
    cands.append(clip(child))

# 2) PCA crossover (5): blend PF pair coefficients
for _ in range(5):
    i,j = rng.choice(nb,2,replace=False)
    ci = (base[i]-mean)@P; cj=(base[j]-mean)@P
    t = rng.uniform(0.3,0.7)
    cchild = ci*(1-t)+cj*t
    child = mean + P@cchild + rng.normal(0,0.01,n_var)
    cands.append(clip(child))

# 3) MoE saturation breaker (5): force high experts+low topk on random baselines
for _ in range(5):
    b = base[rng.randint(nb)].copy()
    slots = rng.choice(16, rng.randint(6,12), replace=False)
    for s in slots:
        b[65+s] = rng.uniform(0.78,1.0)   # 8 experts
        b[81+s] = rng.uniform(0.0,0.35)   # topk=1
    b += rng.normal(0,0.01,n_var)
    cands.append(clip(b))

# 4) Gap-fill targeting largest PF gap (2)
if len(pf_F) >= 2:
    order = np.argsort(pf_F[:,0])
    dists = [np.linalg.norm(pf_F[order[i+1]]-pf_F[order[i]]) for i in range(len(order)-1)]
    if dists:
        gi = np.argmax(dists)
        a = pf_X[order[gi]]; b = pf_X[order[gi+1]]
        for _ in range(2):
            t = rng.uniform(0.3,0.7)
            child = a*(1-t)+b*t + rng.normal(0,0.02,n_var)
            cands.append(clip(child))

# 5) fill remaining with fresh random-space + MoE diversity forcing
while len(cands) < pop_size:
    x = rng.uniform(0,1,n_var)
    exp_choice = rng.choice([0.4,0.6,0.9],16)
    topk_choice = rng.choice([0.2,0.7],16)
    x[65:81] = exp_choice + rng.normal(0,0.03,16)
    x[81:97] = topk_choice + rng.normal(0,0.03,16)
    cands.append(clip(x))

X = np.array(cands[:pop_size])
RESULT = X