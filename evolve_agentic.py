import os
import sys

# Load local .env file from project root if present to populate GOOGLE_API_KEY and model variables
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# Default model variable fallback
if "GEMINI_MODEL" not in os.environ:
    os.environ["GEMINI_MODEL"] = "gemini-3.5-flash"

import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup local agentic-optimizer path
sys.path.insert(0, "/home/stelios/repos/agentic-optimizer")
from agentic_optimizer.metis_agent import MetisAgent

from rwnn.graph import RWNNGraph
from rwnn.mutator import get_gpt2_dag, is_valid_dag

# ---- Continuous-encoding dimensionality ----
# 0: depth | 1-16 attn | 17-32 mlp | 33-48 activation | 49-64 skip
# 65-80: MoE n_experts | 81-96: MoE top_k   (MoE genes default 0 -> E=1 dense = GPT-2)
N_VAR = 97


def multilayer_graph_to_vector(model_type):
    """Encodes standard configurations ('gpt2', 'gpt2-medium', etc.) into an N_VAR continuous vector.
    MoE genes (65:97) are left at 0 -> every FFN is a dense GPT-2 block (backward compatible)."""
    x = np.zeros(N_VAR)
    if model_type == 'gpt2':
        x[0] = 6.0 / 24.0      # 12 layers: 6 + 6 = 12
        x[1:17] = 1.0          # Attention active
        x[17:33] = 1.0         # MLP active
        x[33:49] = 0.1         # GELU activation
        x[49:65] = 0.0         # No skips
    elif model_type == 'gpt2-medium':
        x[0] = 18.0 / 24.0     # 24 layers: 6 + 18 = 24
        x[1:17] = 1.0
        x[17:33] = 1.0
        x[33:49] = 0.1
        x[49:65] = 0.0
    elif model_type == 'gpt2-large':
        x[0] = 1.0             # 30 layers: 6 + 24 = 30
        x[1:17] = 1.0
        x[17:33] = 1.0
        x[33:49] = 0.1
        x[49:65] = 0.0
    else: # Alternating sparse medium (gpt2-sparse)
        x[0] = 18.0 / 24.0     # 24 layers
        x[1:17] = [1.0, 0.0] * 8
        x[17:33] = [0.0, 1.0] * 8
        x[33:49] = 0.1
        x[49:65] = 0.0
    return x


def _decode_experts(e_val):
    """MoE n_experts gene -> {1, 2, 4, 8}. 1 == dense FFN."""
    if e_val < 0.25: return 1
    if e_val < 0.50: return 2
    if e_val < 0.75: return 4
    return 8


def _decode_topk(k_val, n_experts):
    """MoE top_k gene -> {1, 2}, clamped to <= n_experts."""
    k = 1 if k_val < 0.5 else 2
    return min(k, n_experts)


def vector_to_multilayer_graph(x, vocab_size=50257, block_size=256, d_model=768):
    """Decodes a continuous vector x of shape [N_VAR] into a multi-layer H-DAG.
    FFN slots become an MoE block when the n_experts gene decodes to E>1, else a dense FFN."""
    x = np.asarray(x)
    L_var = x[0]
    attn_vars = x[1:17]
    mlp_vars = x[17:33]
    act_vars = x[33:49]
    skip_vars = x[49:65]
    nexp_vars = x[65:81] if x.shape[0] >= 81 else np.zeros(16)   # MoE n_experts genes
    topk_vars = x[81:97] if x.shape[0] >= 97 else np.zeros(16)   # MoE top_k genes

    n_layer = 6 + int(L_var * 24) # Map to [6, 30] layers
    n_layer = min(n_layer, 30)
    
    nodes = [
        {'id': 0, 'type': 'input', 'kwargs': {}},
        {'id': 1, 'type': 'token_embedding', 'kwargs': {'vocab_size': vocab_size, 'd_model': d_model}},
        {'id': 2, 'type': 'positional_embedding', 'kwargs': {'max_seq_len': block_size, 'd_model': d_model}},
        {'id': 3, 'type': 'sum', 'kwargs': {}}
    ]
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3)
    ]
    
    current_x = 3
    layer_out = {}          # residual-stream output node id at the end of each layer
    layer_final_sum = {}    # each active layer's terminal residual SumNode (a valid skip target)
    active_layers = []      # layers that actually instantiate a block
    for l in range(n_layer):
        idx_16 = min(int(l * 16 / n_layer), 15)
        
        ln1_id = 4 + l * 10
        attn_id = 5 + l * 10
        sum_attn_id = 6 + l * 10
        ln2_id = 7 + l * 10
        mlp_up_id = 8 + l * 10
        act_id = 9 + l * 10
        mlp_down_id = 10 + l * 10
        sum_mlp_id = 11 + l * 10
        
        # 1. Causal Attention Sub-Block (active if attn_vars[idx_16] > 0.5)
        if attn_vars[idx_16] > 0.5:
            nodes.append({'id': ln1_id, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}})
            nodes.append({'id': attn_id, 'type': 'causal_attention', 'kwargs': {'n_head': 12, 'd_model': d_model, 'dropout': 0.1}})
            nodes.append({'id': sum_attn_id, 'type': 'sum', 'kwargs': {}})
            
            edges.append((current_x, ln1_id))
            edges.append((ln1_id, attn_id))
            edges.append((current_x, sum_attn_id)) # Residual
            edges.append((attn_id, sum_attn_id))
            current_attn_out = sum_attn_id
        else:
            current_attn_out = current_x
            
        # 2. FFN Sub-Block (active if mlp_vars[idx_16] > 0.5). Dense FFN or MoE per the E gene.
        if mlp_vars[idx_16] > 0.5:
            act_val = act_vars[idx_16]
            if act_val < 0.33:
                act_type = 'gelu'
            elif act_val < 0.66:
                act_type = 'silu'
            else:
                act_type = 'relu'

            n_experts = _decode_experts(nexp_vars[idx_16])
            nodes.append({'id': ln2_id, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}})
            if n_experts > 1:
                # MoE FFN: one fused node (router + experts + top-k combine) replaces up/act/down
                top_k = _decode_topk(topk_vars[idx_16], n_experts)
                nodes.append({'id': mlp_up_id, 'type': 'moe_ffn',
                              'kwargs': {'d_model': d_model, 'n_experts': n_experts, 'top_k': top_k,
                                         'd_hidden': 4 * d_model, 'act_type': act_type, 'dropout': 0.1}})
                nodes.append({'id': sum_mlp_id, 'type': 'sum', 'kwargs': {}})
                edges.append((current_attn_out, ln2_id))
                edges.append((ln2_id, mlp_up_id))
                edges.append((current_attn_out, sum_mlp_id))   # Residual
                edges.append((mlp_up_id, sum_mlp_id))
            else:
                # Dense FFN (standard GPT-2 block)
                nodes.append({'id': mlp_up_id, 'type': 'linear', 'kwargs': {'d_in': d_model, 'd_out': 4 * d_model}})
                nodes.append({'id': act_id, 'type': 'activation', 'kwargs': {'act_type': act_type}})
                nodes.append({'id': mlp_down_id, 'type': 'linear', 'kwargs': {'d_in': 4 * d_model, 'd_out': d_model}})
                nodes.append({'id': sum_mlp_id, 'type': 'sum', 'kwargs': {}})
                edges.append((current_attn_out, ln2_id))
                edges.append((ln2_id, mlp_up_id))
                edges.append((mlp_up_id, act_id))
                edges.append((act_id, mlp_down_id))
                edges.append((current_attn_out, sum_mlp_id))   # Residual
                edges.append((mlp_down_id, sum_mlp_id))
            current_mlp_out = sum_mlp_id
        else:
            current_mlp_out = current_attn_out
            
        # Record this layer's terminal residual SumNode. This is the ONLY correct target
        # for a skip connection: a SumNode adds all of its inputs, so feeding it an earlier
        # block's output produces a genuine residual add. (Prefer the MLP residual sum;
        # fall back to the attention residual sum if the MLP sub-block was bypassed.)
        if mlp_vars[idx_16] > 0.5:
            layer_final_sum[l] = sum_mlp_id
        elif attn_vars[idx_16] > 0.5:
            layer_final_sum[l] = sum_attn_id
        if l in layer_final_sum:
            active_layers.append(l)

        current_x = current_mlp_out
        layer_out[l] = current_x

    # 3. Real cross-layer skip connections (genuine long-range residuals).
    #    A skip re-adds an earlier active block's output into the residual stream of the
    #    active block two active-steps later, by feeding that block's terminal SumNode.
    #    Targeting a SumNode (which sums ALL inputs) makes this a true residual add that
    #    jumps over an intervening active block -- unlike the previous version, which fed a
    #    LayerNormNode (that ignores inputs[1:]) at a fixed nominal offset that rarely existed.
    for i, l in enumerate(active_layers):
        idx_16 = min(int(l * 16 / n_layer), 15)
        if skip_vars[idx_16] > 0.5 and i + 2 < len(active_layers):
            tgt_l = active_layers[i + 2]
            edges.append((layer_out[l], layer_final_sum[tgt_l]))

    # Output Head
    ln_f_id = 4 + n_layer * 10
    head_id = 13
    
    nodes.append({'id': ln_f_id, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}})
    nodes.append({'id': head_id, 'type': 'linear', 'kwargs': {'d_in': d_model, 'd_out': vocab_size}})
    
    edges.append((current_x, ln_f_id))
    edges.append((ln_f_id, head_id))
    
    # Global Defensive Edge Filtering: Ensure both endpoints u and v exist in the nodes set
    node_ids = {n['id'] for n in nodes}
    edges = [(u, v) for u, v in edges if u in node_ids and v in node_ids]
    
    return nodes, edges


def train_and_eval_bpe_model(nodes, edges, d_model=192, max_iters=1000, batch_size=32, block_size=256,
                             parent_state_dict=None, aux_coef=0.01):
    """Trains a compiled H-DAG model on BPE tokens. Returns (model, val_loss, peak_mem_bytes).
    Adds an MoE load-balance aux loss so routers don't collapse to a single expert."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # Load compiled BPE datasets
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')

    model = RWNNGraph(nodes, edges, global_d_model=d_model)
    
    # Lamarckian Weight Inheritance from Parents
    if parent_state_dict is not None:
        copied_keys_count = 0
        child_state = model.state_dict()
        for k, v in parent_state_dict.items():
            if k in child_state and child_state[k].shape == v.shape:
                child_state[k].copy_(v)
                copied_keys_count += 1
        if copied_keys_count > 0:
            print(f" -> Inherited {copied_keys_count} parameter tensors from parent weights.")
            
    model.to(device)

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((d[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((d[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # Training Loop
    model.train()
    for step in range(max_iters):
        xb, yb = get_batch('train')
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss = loss + aux_coef * model.moe_aux_loss()   # MoE load-balancing (0 if no MoE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Evaluate Val Loss (cross-entropy only; aux loss is a training regulariser)
    model.eval()
    val_losses = []
    with torch.no_grad():
        for k in range(5):
            X, Y = get_batch('val')
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            val_losses.append(loss.item())

    peak_mem = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    return model, np.mean(val_losses), peak_mem


# ---- gene grouping (used for the initial population and saturation diagnostics) ----
GENE_ATTN    = list(range(1, 17))
GENE_MLP     = list(range(17, 33))
GENE_ACT     = list(range(33, 49))
GENE_SKIP    = list(range(49, 65))
GENE_NEXPERT = list(range(65, 81))   # MoE n_experts
GENE_TOPK    = list(range(81, 97))   # MoE top_k


def _phenotype(gene_idx, val):
    """Map a raw gene value to the discrete phenotype the decoder actually sees."""
    if gene_idx in GENE_ACT:
        return 'gelu' if val < 0.33 else ('silu' if val < 0.66 else 'relu')
    if gene_idx in GENE_NEXPERT:
        return _decode_experts(val)         # 1 / 2 / 4 / 8 experts
    if gene_idx in GENE_TOPK:
        return 1 if val < 0.5 else 2
    if gene_idx == 0:
        return 6 + min(int(val * 24), 24)   # decoded layer count
    return val > 0.5                        # attn / mlp / skip on-off


def _gene_label(j):
    if j == 0: return "depth"
    if j in GENE_ATTN:    return f"attn[{j-1}]"
    if j in GENE_MLP:     return f"mlp[{j-17}]"
    if j in GENE_ACT:     return f"act[{j-33}]"
    if j in GENE_SKIP:    return f"skip[{j-49}]"
    if j in GENE_NEXPERT: return f"n_experts[{j-65}]"
    return f"top_k[{j-81}]"


def detect_saturated_genes(X):
    """Return {gene_idx: phenotype} for every gene whose phenotype is identical across ALL individuals."""
    X = np.asarray(X)
    sat = {}
    for j in range(X.shape[1]):
        phenos = {_phenotype(j, X[i, j]) for i in range(X.shape[0])}
        if len(phenos) == 1:
            sat[j] = next(iter(phenos))
    return sat


def desaturate_value(gene_idx, pheno, rng):
    """Return a raw value whose phenotype DIFFERS from the saturated one, crossing the plateau threshold."""
    if gene_idx in GENE_ACT:
        others = [v for v, name in [(0.15, 'gelu'), (0.5, 'silu'), (0.85, 'relu')] if name != pheno]
        return float(rng.choice(others))
    if gene_idx in GENE_NEXPERT:
        # jump to a different expert bucket (bucket centers for 1/2/4/8)
        others = [v for v, e in [(0.1, 1), (0.4, 2), (0.6, 4), (0.9, 8)] if e != pheno]
        return float(rng.choice(others))
    if gene_idx in GENE_TOPK:
        return float(rng.uniform(0.6, 0.95)) if pheno == 1 else float(rng.uniform(0.05, 0.4))
    if gene_idx == 0:
        return float(rng.uniform(0.1, 0.95))
    # binary attn/mlp/skip: flip across the 0.5 threshold
    return float(rng.uniform(0.6, 0.95)) if pheno is False else float(rng.uniform(0.05, 0.4))


def active_flops_per_token(nodes, block_size, d_model):
    """Objective 2: active (used) FLOPs per token through the transformer body.
    Attention + FFN only (embeddings and the final head are constant across architectures
    and excluded). A dense FFN counts its full cost; an MoE FFN counts only its top_k
    active experts (+ router) -- so extra experts add capacity/params/memory but NOT FLOPs,
    which is exactly what makes MoE attractive under this objective."""
    T = block_size
    flops = 0.0
    for n in nodes:
        t = n['type']
        if t == 'causal_attention':
            d = n['kwargs']['d_model']
            flops += 2 * (4 * d * d)        # qkv + out projections (mac*2)
            flops += 2 * (2 * T * d)        # QK^T and attn*V (mac*2)
        elif t == 'linear':
            if n['id'] == 13:               # exclude the LM head (constant)
                continue
            flops += 2 * (n['kwargs']['d_in'] * n['kwargs']['d_out'])
        elif t == 'moe_ffn':
            kw = n['kwargs']
            d, h, E, topk = kw['d_model'], kw['d_hidden'], kw['n_experts'], kw['top_k']
            flops += 2 * (d * E)            # router
            flops += topk * 2 * (2 * d * h) # only top_k active experts (each expert = d*h + h*d macs)
    return flops


def build_initial_population(pop_size, seed=1234):
    """5 GPT-2-family seeds (incl. the 2 first-elite-set vectors) + (pop_size-5) diverse explorers
    that use real skip connections, front-loaded attention, non-uniform MLP, and mixed activations."""
    rng = np.random.RandomState(seed)
    pop = []

    def _pad(v):
        """Pad/truncate a vector to N_VAR (MoE genes default 0 -> dense)."""
        v = np.asarray(v, dtype=float)
        if v.shape[0] < N_VAR:
            v = np.concatenate([v, np.zeros(N_VAR - v.shape[0])])
        return v[:N_VAR]

    # --- 5 GPT-2 family seeds (the 2 first-elite-set vectors first, if available) ---
    if os.path.exists("seed_elites.json"):
        for e in json.load(open("seed_elites.json"))[:2]:
            pop.append(_pad(e['vector']))                    # 65-D elites padded to N_VAR (dense)
    pop.append(multilayer_graph_to_vector('gpt2'))
    pop.append(multilayer_graph_to_vector('gpt2-sparse'))
    # 5th variant: a dense 18-layer GPT-2-style stack (~185M). Distinct depth, and safely under
    # the 12 GB OOM ceiling -- replaces gpt2-large (30L / ~290M) which OOMs during training.
    gpt2_18 = np.zeros(N_VAR)
    gpt2_18[0] = 0.5          # 6 + int(0.5*24) = 18 layers
    gpt2_18[1:33] = 1.0       # every layer: attention + MLP both active
    gpt2_18[33:49] = 0.1      # GELU
    gpt2_18[49:65] = 0.0      # no skips (65:97 stay 0 -> dense)
    pop.append(gpt2_18)
    while len(pop) < 5:                                   # fallback if seed_elites.json was missing
        pop.append(multilayer_graph_to_vector('gpt2-medium'))
    pop = pop[:5]

    # --- (pop_size - 5) structurally-diverse, UNBIASED explorers ---
    n_unbiased = max(0, pop_size - 5)
    for i in range(n_unbiased):
        x = np.zeros(N_VAR)
        x[0] = np.clip((i + 0.5) / n_unbiased, 0.12, 0.95)     # spread depth / model size

        # FRONT-LOADED attention: dominant in early slots, tapering later
        front = rng.randint(4, 11)
        for s in range(16):
            base = 0.85 if s < front else 0.25
            x[1 + s] = np.clip(base + rng.normal(0, 0.12), 0.0, 1.0)

        # NON-UNIFORM MLP placement (per-individual density)
        mlp_density = rng.uniform(0.35, 0.8)
        for s in range(16):
            on = rng.rand() < mlp_density
            x[17 + s] = np.clip((0.8 if on else 0.2) + rng.normal(0, 0.1), 0.0, 1.0)

        # VARIED activations: bias each individual toward a different family, one fully mixed
        style = i % 4
        for s in range(16):
            if style == 0:   v = rng.uniform(0.00, 0.30)   # GELU-leaning
            elif style == 1: v = rng.uniform(0.36, 0.63)   # SiLU-leaning
            elif style == 2: v = rng.uniform(0.70, 1.00)   # ReLU-leaning
            else:            v = rng.uniform(0.00, 1.00)    # fully mixed
            x[33 + s] = v

        # SKIP CONNECTIONS present (per-individual density), with a guaranteed minimum
        skip_density = rng.uniform(0.4, 0.85)
        for s in range(16):
            x[49 + s] = np.clip((0.8 if rng.rand() < skip_density else 0.2) + rng.normal(0, 0.1), 0.0, 1.0)
        for s in rng.choice(16, size=3, replace=False):
            x[49 + s] = rng.uniform(0.6, 0.95)

        # MoE GENES: give ~half the explorers real MoE variance so n_experts/top_k are not
        # born saturated at dense (E=1). Under the active-FLOPs objective, MoE = capacity for free.
        if i % 2 == 0:
            # dense explorer: leave n_experts genes low (E=1) but keep small noise for variance
            x[65:81] = rng.uniform(0.0, 0.24, 16)   # -> E=1
            x[81:97] = rng.uniform(0.0, 1.0, 16)
        else:
            # MoE explorer: per-slot varied expert counts (2/4/8) and top_k (1/2)
            e_center = rng.choice([0.4, 0.6, 0.9])  # bias this individual toward 2 / 4 / 8 experts
            for s in range(16):
                x[65 + s] = np.clip(e_center + rng.normal(0, 0.12), 0.26, 1.0)   # E in {2,4,8}
                x[81 + s] = rng.uniform(0.0, 1.0)                                 # top_k in {1,2}

        pop.append(np.clip(x, 0.0, 1.0))

    return np.array(pop[:pop_size])


def run_agentic_optimization(generations=10, pop_size=20, eval_steps=57860, use_lamarckian=True):
    print("=== AGENTIC OPTIMIZATION LOOP STARTED ===")
    
    # Forcefully clear old agentic-optim directory for a fresh big-model run
    checkpoint_dir = "checkpoints/agentic-optim"
    if os.path.exists(checkpoint_dir):
        print(f"Clearing old agentic-optim directory: {checkpoint_dir}")
        import shutil
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    vocab_size = 50257
    block_size = 256
    d_model = 768  # Scaled up d_model to support 100M - 500M parameter models
    highest_gen = 0
    all_historical_pareto_F = []

    # Build the initial population: 5 GPT-2-family seeds (including the 2 first-elite-set
    # vectors from seed_elites.json) + (pop_size-5) structurally-diverse, UNBIASED explorers
    # that actively use skip connections, front-loaded attention, non-uniform MLP placement
    # and a mix of activations. This deliberately breaks the old all-GELU / no-skip init bias.
    initial_pop_arr = build_initial_population(pop_size)
    print(f"Initialized population of {len(initial_pop_arr)}: 5 GPT-2 variants + {max(0, pop_size-5)} diverse explorers.")

    # 2. Plain-Text Auxiliary Context (Domain Knowledge Injection)
    problem_context = (
        "You are optimizing a multi-layer stacked Randomly Wired Large Language Model (RWNN-LLM) H-DAG.\n\n"
        "## Decision Variable Guide (x is of size 97):\n"
        "- Variable 0 controls the depth of the model (number of layers, from 6 to 30 layers).\n"
        "- Variables 1 to 16: attention active (> 0.5) or bypassed (<= 0.5) per layer-slot.\n"
        "- Variables 17 to 32: feed-forward (FFN) active (> 0.5) or bypassed (<= 0.5) per layer-slot.\n"
        "- Variables 33 to 48: activation type ([0,0.33] GELU, [0.33,0.66] SiLU, [0.66,1.0] ReLU).\n"
        "- Variables 49 to 64 (> 0.5): add a genuine long-range residual skip (block output re-added two active-layers later).\n"
        "- Variables 65 to 80: MoE n_experts per FFN slot ([0,0.25]->1 dense, [0.25,0.5]->2, [0.5,0.75]->4, [0.75,1]->8 experts).\n"
        "- Variables 81 to 96: MoE top_k per FFN slot (<0.5 -> 1 active expert, >=0.5 -> 2), clamped to n_experts.\n\n"
        "## Domain Knowledge & Structural Physics:\n"
        "1. Active Depth: Deeper models converge faster and reach lower validation loss, but cost more FLOPs and memory.\n"
        "2. Bypass Channels: Bypassing attention or FFN at some layers cuts FLOPs and memory with a small loss penalty.\n"
        "3. Cross-Layer Skips: A skip re-adds an earlier block's output into a later block's residual stream (DenseNet-style), improving gradient flow at almost no FLOP cost.\n"
        "4. Activations: Mix GELU/SiLU/ReLU across layers; do not leave every layer on one activation.\n"
        "5. Front-Loaded Attention: Attention EARLY, FFN LATER ('sandwich' ordering) tends to lower loss at equal budget.\n"
        "6. MIXTURE-OF-EXPERTS (KEY): An FFN slot with n_experts>1 becomes an MoE block. Only top_k experts run per token, so **extra experts add capacity (lower loss) and parameters/memory but almost NO active-FLOPs**. Because Objective 2 is active-FLOPs (not parameters), MoE lets you buy loss reduction very cheaply -- exploit it. n_experts controls capacity; top_k controls active-FLOPs; both are bounded by the memory constraint below.\n\n"
        "## Objectives (minimize both):\n"
        "- Objective 1: Validation Cross-Entropy Loss.\n"
        "- Objective 2: Active-FLOPs per token (attention + only the top_k active experts of each FFN). Total parameters are NOT an objective -- only used compute is.\n\n"
        "## Hard Constraints (a violating candidate is REJECTED, not scored):\n"
        "- Peak training memory must fit the 12 GB GPU budget (large n_experts inflates memory even though FLOPs stay low -- this is what bounds MoE size).\n"
        "- Validation loss must be < 4.5 (kills degenerate near-empty models).\n"
        "- At least 3 active blocks (no empty / embeddings-only models).\n"
        "- Reference point for hypervolume is (6.0e9 active-FLOPs, 4.5 loss).\n\n"
        "## Search Guidance:\n"
        "- Keep the population DIVERSE across ALL gene groups, especially the MoE genes (65-96): try n_experts in {2,4,8} with top_k in {1,2}, not just dense.\n"
        "- The sweet spot is HIGH n_experts + LOW top_k (big capacity, low active-FLOPs) pushed until the memory constraint bites.\n"
        "- SATURATION: if any variable collapses to one value across the whole population for two+ generations it is stuck on a flat plateau; dedicate a couple of exploration individuals to markedly DIFFERENT values that CROSS its thresholds."
    )

    # Objective/constraint configuration
    LOSS_MAX = 4.5            # hard loss constraint (kills degenerate empty models)
    MIN_ACTIVE_BLOCKS = 3     # hard structural constraint
    MEM_BUDGET = 11.0e9       # hard peak-memory constraint (bytes) on the 12 GB card
    PENALTY_LOSS = 10.0       # objective-1 value assigned to a constraint-violating candidate

    # 3. Instantiate MetisAgent  (Obj1 = loss, Obj2 = active-FLOPs/token)
    agent = MetisAgent(
        n_var=N_VAR,
        n_obj=2,
        bounds=[(0.0, 1.0)] * N_VAR,
        population_size=pop_size,
        initial_population=initial_pop_arr,
        max_elites=100,
        problem_context=problem_context
    )
    # Reference point for Hypervolume (Rx = 6.0e9 active-FLOPs, Ry = 4.5 loss)
    agent.ref = [4.5, 6.0e9]
    
    # Adjust starting generation and restore Agent's internal state memory if resuming
    if highest_gen > 0:
        agent.generation = highest_gen
        
        # Populate agent's Pareto front and evaluated memory so context builders and diagnostics run smoothly
        pf_X_list = []
        pf_F_list = []
        report_file = f"checkpoints/agentic-optim/generation_{highest_gen}_report.json"
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r') as f:
                    elites = json.load(f)
                for e in elites:
                    config_file = e['saved_config']
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            c_data = json.load(f)
                        pf_X_list.append(np.array(c_data['vector']))
                        pf_F_list.append(np.array([c_data['loss'], c_data['params']]))
            except Exception as ex:
                print(f"Failed to restore agent memory from reports: {ex}")
                
        agent.pf_X = pf_X_list
        agent.pf_F = pf_F_list
        agent._all_X = pf_X_list
        agent._all_F = pf_F_list

    # 4. Restore state dicts memory map from previous weight files if resuming
    X_hash_to_state = {}
    if highest_gen > 0:
        for i, elite in enumerate(pf_X_list):
            loss_val = pf_F_list[i][0]
            weight_file = f"checkpoints/agentic-optim/pareto_gen{highest_gen}_ind{i+1}_loss{loss_val:.2f}.pt"
            if os.path.exists(weight_file):
                try:
                    X_hash_to_state[tuple(elite)] = torch.load(weight_file, map_location='cpu')
                except Exception as ex:
                    print(f"Failed to load weight file {weight_file} into memory: {ex}")

    # Saturation diagnostics: track genes whose phenotype is frozen across the whole population.
    saturation_streak = {}      # gene index -> consecutive generations fully saturated
    genes_to_desaturate = {}    # genes (flagged last gen) to break in the NEXT generation
    SATURATION_GENS = 2         # "a couple of generations"
    desat_rng = np.random.RandomState(4321)

    # Plotting history: cumulative cloud of ALL feasible evaluated individuals + the Gen-1 GPT-2 seeds
    all_eval_points = []        # [[loss, active_flops], ...] across every generation
    gpt2_seed_points = []       # [(active_flops, loss, label), ...] captured in generation 1

    # 5. Agentic Optimization Loop
    for gen in range(highest_gen, generations):
        print(f"\n--- Agentic Generation {gen + 1} / {generations} ---")
        
        # Agent writes Python code and proposes candidates
        X = agent.ask()
        
        # Allow a small number of elites to be promoted to the next generation just to continue their fine-tuning
        num_promoted_elites = 2
        if gen > 0 and len(agent.pf_X) > 0:
            # Sort elites by loss (ascending)
            sorted_elites_indices = np.argsort([f[0] for f in agent.pf_F])
            top_elites_X = [agent.pf_X[idx] for idx in sorted_elites_indices[:num_promoted_elites]]
            
            # Replace the last candidates in X with our top elites
            for i, elite_vec in enumerate(top_elites_X):
                if i < len(X):
                    X[-(i+1)] = elite_vec.copy()
            print(f" -> Promoted {len(top_elites_X)} elites directly to the population to continue their fine-tuning.")

        # Break saturation: force markedly different values into 2 exploration individuals.
        # Slots 0-1 are agent-proposed explorers (elites were promoted into the LAST slots).
        if genes_to_desaturate and pop_size >= 4:
            for ei in (0, 1):
                for j, pheno in genes_to_desaturate.items():
                    X[ei][j] = desaturate_value(j, pheno, desat_rng)
            print(f" -> Broke saturation: injected diverse values into individuals 0,1 for "
                  f"{len(genes_to_desaturate)} gene(s): {', '.join(_gene_label(j) for j in genes_to_desaturate)}")

        # Save generated sampling code
        code_file = f"checkpoints/agentic-optim/generation_{gen+1}_code.py"
        with open(code_file, 'w') as f:
            f.write(agent.last_code)
        print(f"✓ Saved generated sampling code: {code_file}")

        # Batch evaluation. Objectives = [loss, active_flops]; constraints = memory, loss<4.5, min blocks.
        F = []
        state_dicts = []
        meta = []   # per-candidate diagnostics for reporting
        for idx in range(pop_size):
            x = X[idx]
            nodes, edges = vector_to_multilayer_graph(x, vocab_size, block_size, d_model=d_model)

            # Structure metrics (no training needed)
            dummy_model = RWNNGraph(nodes, edges, global_d_model=d_model)
            params = sum(p.numel() for p in dummy_model.parameters())
            active_flops = active_flops_per_token(nodes, block_size, d_model)
            active_blocks = len({(n['id'] - 4) // 10 for n in nodes
                                 if n['id'] >= 4 and n['id'] != 13 and (n['id'] - 4) % 10 in (1, 4)})
            moe_nodes = [n for n in nodes if n['type'] == 'moe_ffn']
            total_experts = sum(n['kwargs']['n_experts'] for n in moe_nodes)
            moe_tag = f" MoE[{len(moe_nodes)} blk/{total_experts} exp]" if moe_nodes else ""

            # Distance-Based Ancestry Matching for Lamarckian Weight Inheritance / Resume Training
            parent_state = None
            if len(agent.pf_X) > 0:
                Xp_arr = np.array(agent.pf_X)
                dists = np.linalg.norm(Xp_arr - x, axis=1)
                best_idx = int(np.argmin(dists)); min_dist = dists[best_idx]
                if min_dist < 1e-5:
                    print(f" -> Promoted elite exact match. Resuming from its previous checkpoint.")
                    parent_state = X_hash_to_state.get(tuple(agent.pf_X[best_idx]))
                elif use_lamarckian and min_dist < 0.6:
                    parent_state = X_hash_to_state.get(tuple(agent.pf_X[best_idx]))

            print(f"Evaluating candidate {idx+1}/{pop_size} (Params: {params:,}, "
                  f"aFLOPs/tok: {active_flops/1e9:.3f}G, blocks: {active_blocks}{moe_tag})...")

            feasible, reason, peak_mem = True, "ok", 0
            # Constraint: minimum active blocks (cheap pre-check, skip training)
            if active_blocks < MIN_ACTIVE_BLOCKS:
                feasible, reason, val_loss, s_dict = False, "too_few_blocks", PENALTY_LOSS, None
                print(f" -> REJECTED: only {active_blocks} active blocks (< {MIN_ACTIVE_BLOCKS}).")
            else:
                try:
                    import gc
                    gc.collect(); torch.cuda.empty_cache()
                    trained_model, val_loss, peak_mem = train_and_eval_bpe_model(
                        nodes, edges, d_model=d_model, max_iters=eval_steps, batch_size=8,
                        block_size=block_size, parent_state_dict=parent_state)
                    print(f" -> Success! Val Loss: {val_loss:.4f} | peak mem: {peak_mem/1e9:.2f} GB")
                    if peak_mem > MEM_BUDGET:
                        feasible, reason, s_dict = False, "memory", None
                        print(f" -> REJECTED: peak memory {peak_mem/1e9:.2f} GB > budget {MEM_BUDGET/1e9:.1f} GB.")
                        val_loss = PENALTY_LOSS
                    elif val_loss >= LOSS_MAX:
                        feasible, reason, s_dict = False, "loss", None
                        print(f" -> REJECTED: loss {val_loss:.4f} >= {LOSS_MAX}.")
                    else:
                        s_dict = trained_model.state_dict()
                    del trained_model
                    del dummy_model
                    gc.collect(); torch.cuda.empty_cache()
                except Exception as e:
                    feasible, reason, val_loss, s_dict = False, "oom", PENALTY_LOSS, None
                    print(f" -> REJECTED (constraint): {str(e)[:80]}")

            F.append([val_loss, active_flops])
            state_dicts.append(s_dict)
            meta.append({'params': int(params), 'active_flops': float(active_flops),
                         'active_blocks': int(active_blocks), 'peak_mem': int(peak_mem),
                         'n_moe_blocks': len(moe_nodes), 'total_experts': int(total_experts),
                         'feasible': bool(feasible), 'reason': reason})

        F_arr = np.array(F)

        # Cache newly evaluated state-dicts
        for k in range(pop_size):
            if state_dicts[k] is not None:
                X_hash_to_state[tuple(X[k])] = state_dicts[k]

        n_feasible = sum(1 for m in meta if m['feasible'])
        print(f" -> {n_feasible}/{pop_size} feasible; rejected reasons: "
              f"{ {r: sum(1 for m in meta if m['reason']==r) for r in set(m['reason'] for m in meta if not m['feasible'])} }")

        # Accumulate the cumulative cloud (all feasible evaluated individuals) and, in generation 1,
        # capture the 5 dense GPT-2 seed architectures (first 5 of the initial population) to annotate.
        for k in range(pop_size):
            if F_arr[k][0] < LOSS_MAX:
                all_eval_points.append([float(F_arr[k][0]), float(F_arr[k][1])])
        if gen == 0:
            seed_labels = ["gpt2-medium (24L)", "15L", "gpt2-small (12L)", "gpt2-sparse (24L)", "18L dense"]
            for k in range(min(5, pop_size)):
                if F_arr[k][0] < LOSS_MAX:
                    gpt2_seed_points.append((float(F_arr[k][1]), float(F_arr[k][0]), seed_labels[k]))

        # Dynamic hypervolume reference point (Obj2 is now active-FLOPs)
        combined_F = list(all_historical_pareto_F) + [f for f in F_arr.tolist() if f[0] < LOSS_MAX]
        if len(combined_F) > 0:
            arr = np.array(combined_F)
            agent.ref = [max(1.1 * float(np.max(arr[:, 0])), LOSS_MAX), 1.1 * float(np.max(arr[:, 1]))]
            print(f" -> Dynamic Reference Point: Loss={agent.ref[0]:.4f}, aFLOPs={agent.ref[1]/1e9:.3f}G")
            
        # Report results back to MetisAgent
        agent.tell(X, F_arr)

        # --- Saturation detection on the evaluated population (report + arm de-saturation) ---
        sat = detect_saturated_genes(X)
        for j in list(saturation_streak):
            if j not in sat:
                del saturation_streak[j]
        for j in sat:
            saturation_streak[j] = saturation_streak.get(j, 0) + 1
        genes_to_desaturate = {j: sat[j] for j, n in saturation_streak.items() if n >= SATURATION_GENS}
        if genes_to_desaturate:
            details = ", ".join(f"{_gene_label(j)}={genes_to_desaturate[j]}[{saturation_streak[j]}g]"
                                for j in genes_to_desaturate)
            print(f" ⚠ SATURATION across all {pop_size} individuals for >= {SATURATION_GENS} gens: {details}")
            print(f"    -> will force diverse values into 2 explorers next generation.")

        # Get Pareto Front elites, restricted to FEASIBLE ones (loss < LOSS_MAX).
        Xp, Fp = agent.result()
        valid_idx = [i for i, f in enumerate(Fp) if f[0] < LOSS_MAX]
        Xp = Xp[valid_idx] if len(valid_idx) > 0 else Xp
        Fp = Fp[valid_idx] if len(valid_idx) > 0 else Fp

        # Update historical Pareto front members list
        all_historical_pareto_F = [f for f in Fp]

        # Prune cached state dicts to active Pareto members only
        active_keys = set(tuple(x) for x in Xp)
        X_hash_to_state = {k: v for k, v in X_hash_to_state.items() if k in active_keys}

        print(f"\n🏆 Generation {gen + 1} Agentic Pareto Front (Obj1=loss, Obj2=active-FLOPs):")

        pareto_reports = []
        for i, (x_elite, f_elite) in enumerate(zip(Xp, Fp)):
            loss_val = float(f_elite[0])
            flops_val = float(f_elite[1])

            elite_nodes, elite_edges = vector_to_multilayer_graph(x_elite, vocab_size, block_size, d_model=d_model)

            # Look up this elite's per-candidate diagnostics + save weights if it was evaluated this gen
            m = None
            weight_file = "checkpoints/agentic-optim/no_weights.pt"
            for k in range(pop_size):
                if np.array_equal(X[k], x_elite):
                    m = meta[k]
                    if state_dicts[k] is not None:
                        weight_file = f"checkpoints/agentic-optim/pareto_gen{gen+1}_ind{i+1}_loss{loss_val:.2f}.pt"
                        try:
                            torch.save(state_dicts[k], weight_file)
                        except Exception as save_err:
                            print(f" -> Warning: could not save weights ({save_err}).")
                    break
            params_val = m['params'] if m else 0
            moe_str = f", MoE {m['n_moe_blocks']}blk/{m['total_experts']}exp" if (m and m['n_moe_blocks']) else ""
            print(f"  Elite {i+1}: Loss={loss_val:.4f}, aFLOPs/tok={flops_val/1e9:.3f}G, "
                  f"Params={params_val:,}{moe_str}")

            config_file = f"checkpoints/agentic-optim/pareto_gen{gen+1}_ind{i+1}_config.json"
            config_data = {
                'nodes': elite_nodes, 'edges': elite_edges,
                'loss': loss_val, 'active_flops': flops_val, 'params': params_val,
                'active_blocks': (m['active_blocks'] if m else None),
                'peak_mem': (m['peak_mem'] if m else None),
                'n_moe_blocks': (m['n_moe_blocks'] if m else 0),
                'total_experts': (m['total_experts'] if m else 0),
                'vector': x_elite.tolist()
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)

            pareto_reports.append({
                'rank': i + 1, 'nodes': len(elite_nodes), 'edges': len(elite_edges),
                'loss': loss_val, 'active_flops': flops_val, 'params': params_val,
                'active_blocks': (m['active_blocks'] if m else None),
                'n_moe_blocks': (m['n_moe_blocks'] if m else 0),
                'total_experts': (m['total_experts'] if m else 0),
                'saved_weights': weight_file, 'saved_config': config_file
            })

        report_file = f"checkpoints/agentic-optim/generation_{gen+1}_report.json"
        with open(report_file, 'w') as f:
            json.dump(pareto_reports, f, indent=4)
        print(f"✓ Saved {report_file}")

        # Plot: cumulative gray cloud of ALL evaluated individuals + current Pareto front,
        # with the Gen-1 GPT-2 dense seed architectures annotated. x = active-FLOPs, y = loss.
        plt.figure(figsize=(9, 6.5))
        if all_eval_points:
            allF = np.array(all_eval_points)
            plt.scatter(allF[:, 1] / 1e9, allF[:, 0], color='#9aa7b4', alpha=0.45, s=32,
                        label='All evaluated (cumulative)', zorder=2)
        elite_valid = Fp[Fp[:, 0] < LOSS_MAX] if len(Fp) else Fp
        if len(elite_valid) > 0:
            ex = elite_valid[:, 1] / 1e9; ey = elite_valid[:, 0]; order = np.argsort(ex)
            if len(ex) > 1:
                plt.plot(ex[order], ey[order], '--', color='#ff3333', alpha=0.8, zorder=3)
            plt.scatter(ex, ey, color='#ff3333', s=140, marker='*', edgecolor='white',
                        linewidth=0.6, label='Pareto frontier (elites)', zorder=5)
        if gpt2_seed_points:
            sx = [p[0] / 1e9 for p in gpt2_seed_points]; sy = [p[1] for p in gpt2_seed_points]
            plt.scatter(sx, sy, marker='s', s=95, color='#2b6cb0', edgecolor='white',
                        linewidth=0.6, label='GPT-2 seeds (Gen 1)', zorder=6)
            for (fx, ly, lab) in gpt2_seed_points:
                plt.annotate(lab, (fx / 1e9, ly), textcoords='offset points', xytext=(6, 6),
                             fontsize=7.5, color='#1a3a5c', zorder=7)
        plt.xlabel("Active-FLOPs per token (GFLOPs)")
        plt.ylabel("Validation Loss (Cross-Entropy)")
        plt.title(f"Gen {gen+1} Pareto Front — loss vs active-FLOPs (cumulative, MoE-aware)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        plot_file = f"checkpoints/agentic-optim/generation_{gen+1}_pareto.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Pareto plot: {plot_file}")

        # Save Agent learnings and history to disk for blog3 documentation
        if agent._strategy_log:
            last_entry = agent._strategy_log[-1]
            learnings_file = "checkpoints/agentic-optim/agent_learnings_history.txt"
            with open(learnings_file, "a") as f:
                f.write(f"\n======================================== GENERATION {gen+1} ========================================\n")
                f.write(f"Hypervolume: {last_entry['hv']:.4f} (Delta: {last_entry['hv_delta'] if last_entry['hv_delta'] is not None else 0.0:+.4f})\n")
                f.write(f"Strategy: {last_entry['code_summary']}\n")
                f.write(f"Learnings:\n{last_entry['learnings']}\n")
            print(f"✓ Saved Agent learnings to: {learnings_file}")

    print("\n=== AGENTIC OPTIMIZATION LOOP COMPLETE ===")


if __name__ == "__main__":
    run_agentic_optimization(generations=10, pop_size=20, eval_steps=57860, use_lamarckian=True)
