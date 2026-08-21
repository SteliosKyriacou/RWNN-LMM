# Strategy: PCA-EA (fixed) + gap-fill + coordinated MoE-group push + extreme pushing
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

mean = base.mean(0)
centered = base - mean
U,S,Vt = np.linalg.svd(centered, full_matrices=False)
k = min(6, len(S))
P = Vt[:k].T
scale = S[:k].mean()*0.06 + 1e-6

# 1) PCA mutation (6)
for _ in range(6):
    b = base[rng.randint(nb)]
    coef = (b-mean) @ P
    noise = rng.normal(0,1,k) * np.linspace(1.0,0.2,k) * scale
    child = mean + P@(coef+noise)
    cands.append(clip(child))

# 2) PCA crossover (5)
for _ in range(5):
    i,j = rng.choice(nb,2,replace=False)
    ci=(base[i]-mean)@P; cj=(base[j]-mean)@P
    t = rng.uniform(0.3,0.7)
    child = mean + P@(ci*(1-t)+cj*t) + rng.normal(0,0.01,n_var)
    cands.append(clip(child))

# 3) Gap-fill at largest PF gap (3)
if len(pf_F)>=2:
    order = np.argsort(pf_F[:,0])
    dists = [np.linalg.norm(pf_F[order[i+1]]-pf_F[order[i]]) for i in range(len(order)-1)]
    if dists:
        gi = np.argmax(dists)
        a = pf_X[order[gi]]; b = pf_X[order[gi+1]]
        for _ in range(3):
            t = rng.uniform(0.2,0.8)
            child = a*(1-t)+b*t + rng.normal(0,0.02,n_var)
            cands.append(clip(child))

# 4) Coordinated MoE-group push (moderate, not extreme) (3)
for _ in range(3):
    b = base[rng.randint(nb)].copy()
    slots = rng.choice(16, rng.randint(4,8), replace=False)
    for s in slots:
        b[65+s] = np.clip(b[65+s] + rng.uniform(0.1,0.25),0,1)
        b[81+s] = np.clip(b[81+s] - rng.uniform(0.05,0.2),0,1)
    b += rng.normal(0,0.015,n_var)
    cands.append(clip(b))

# 5) Extreme push f2 low (moderate, learn from past failure - smaller magnitude) (2)
for _ in range(2):
    b = base[np.argmin(base_f2 if False else 0)] if False else base[rng.randint(nb)].copy()
    b[65:81] = np.clip(b[65:81] + rng.uniform(0.05,0.15,16),0,1)
    b[81:97] = np.clip(b[81:97] - rng.uniform(0.05,0.15,16),0,1)
    b += rng.normal(0,0.02,n_var)
    cands.append(clip(b))

# 6) fill remaining random with MoE diversity
while len(cands) < pop_size:
    x = rng.uniform(0,1,n_var)
    exp_choice = rng.choice([0.4,0.6,0.9],16)
    topk_choice = rng.choice([0.2,0.7],16)
    x[65:81] = exp_choice + rng.normal(0,0.03,16)
    x[81:97] = topk_choice + rng.normal(0,0.03,16)
    cands.append(clip(x))

X = np.array(cands[:pop_size])
RESULT = X