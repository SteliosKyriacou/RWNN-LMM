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

def multilayer_graph_to_vector(model_type):
    """Encodes standard configurations ('gpt2', 'gpt2-medium', etc.) into a 65-dimensional continuous vector."""
    x = np.zeros(65)
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


def vector_to_multilayer_graph(x, vocab_size=50257, block_size=256, d_model=768):
    """Decodes a continuous vector x of shape [65] back into a fully connected multi-layer H-DAG."""
    L_var = x[0]
    attn_vars = x[1:17]
    mlp_vars = x[17:33]
    act_vars = x[33:49]
    skip_vars = x[49:65]
    
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
            
        # 2. MLP Sub-Block (active if mlp_vars[idx_16] > 0.5)
        if mlp_vars[idx_16] > 0.5:
            act_val = act_vars[idx_16]
            if act_val < 0.33:
                act_type = 'gelu'
            elif act_val < 0.66:
                act_type = 'silu'
            else:
                act_type = 'relu'
                
            nodes.append({'id': ln2_id, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}})
            nodes.append({'id': mlp_up_id, 'type': 'linear', 'kwargs': {'d_in': d_model, 'd_out': 4 * d_model}})
            nodes.append({'id': act_id, 'type': 'activation', 'kwargs': {'act_type': act_type}})
            nodes.append({'id': mlp_down_id, 'type': 'linear', 'kwargs': {'d_in': 4 * d_model, 'd_out': d_model}})
            nodes.append({'id': sum_mlp_id, 'type': 'sum', 'kwargs': {}})
            
            edges.append((current_attn_out, ln2_id))
            edges.append((ln2_id, mlp_up_id))
            edges.append((mlp_up_id, act_id))
            edges.append((act_id, mlp_down_id))
            edges.append((current_attn_out, sum_mlp_id)) # Residual
            edges.append((mlp_down_id, sum_mlp_id))
            current_mlp_out = sum_mlp_id
        else:
            current_mlp_out = current_attn_out
            
        # 3. Cross-layer skip connections
        if l < n_layer - 2 and skip_vars[idx_16] > 0.5:
            target_ln1 = 4 + (l + 2) * 10
            edges.append((current_mlp_out, target_ln1))
            
        current_x = current_mlp_out
        
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


def train_and_eval_bpe_model(nodes, edges, d_model=192, max_iters=1000, batch_size=32, block_size=256, parent_state_dict=None):
    """Trains a compiled H-DAG model on BPE tokens and returns validation loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Evaluate Val Loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for k in range(5):
            X, Y = get_batch('val')
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            val_losses.append(loss.item())
            
    return model, np.mean(val_losses)


def run_agentic_optimization(generations=100, pop_size=10, eval_steps=1000):
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

    initial_population = []
    
    # Seed with standard GPT architectures scaled between 100M and 500M params:
    gpt2_vec = multilayer_graph_to_vector('gpt2')
    gpt2_medium_vec = multilayer_graph_to_vector('gpt2-medium')
    gpt2_large_vec = multilayer_graph_to_vector('gpt2-large')
    gpt2_sparse_vec = multilayer_graph_to_vector('gpt2-sparse')
    
    initial_population.append(gpt2_vec)
    initial_population.append(gpt2_medium_vec)
    initial_population.append(gpt2_large_vec)
    initial_population.append(gpt2_sparse_vec)
    loaded_seeds_count = 4
    
    # Fill up the rest of the initial population
    while len(initial_population) < pop_size:
        # Breed from our previous seeds to populate the remaining slots
        parent_a = random.choice(initial_population[:loaded_seeds_count])
        parent_b = random.choice(initial_population[:loaded_seeds_count])
        # Linear blend mutation for vectors
        blend = np.random.uniform(0.0, 1.0, 65)
        child = blend * parent_a + (1.0 - blend) * parent_b
        if random.random() < 0.3:
            child += np.random.randn(65) * 0.1
        initial_population.append(np.clip(child, 0.0, 1.0))
        
    initial_pop_arr = np.array(initial_population)

    # 2. Plain-Text Auxiliary Context (Domain Knowledge Injection)
    problem_context = (
        "You are optimizing a multi-layer stacked Randomly Wired Large Language Model (RWNN-LLM) H-DAG.\n\n"
        "## Decision Variable Guide (x is of size 65):\n"
        "- Variable 0 controls the depth of the model (number of layers, from 6 to 30 layers).\n"
        "- Variables 1 to 16 represent the fractional attention states active (> 0.5) or bypassed (<= 0.5).\n"
        "- Variables 17 to 32 represent the fractional MLP states active (> 0.5) or bypassed (<= 0.5).\n"
        "- Variables 33 to 48 represent the fractional activation types ([0, 0.33] for GELU, [0.33, 0.66] for SiLU, [0.66, 1.0] for ReLU).\n"
        "- Variables 49 to 64 represent whether to add a cross-layer bypass skip connection.\n\n"
        "## Domain Knowledge & Structural Physics:\n"
        "1. Active Depth: Deeper models (larger Var 0) have more parameters but converge much faster and achieve lower validation loss.\n"
        "2. Bypass Channels: Bypassing attention or MLP at some layers reduces parameter complexity with minimal loss penalty.\n"
        "3. Cross-Layer Skips: Skip connections establish deep residual streams, enabling stable gradient backpropagation.\n"
        "4. Activations: Rotating between GELU, SiLU, and ReLU can dynamically reshape MLP representational capacity.\n\n"
        "## Objectives and Constraints:\n"
        "- Minimizing Objective 1: Validation Cross-Entropy Loss (Perplexity). You must keep loss strictly < 6.0. Any loss >= 6.0 is a complete failure.\n"
        "- Minimizing Objective 2: Trainable parameter count (Complexity).\n"
        "- Maintain a stable trade-off frontier. Deeper networks (with active residual bypasses) will have more parameters but achieve lower loss.\n"
        "- Reference point for hypervolume calculation is (5.0e8 parameters, 6.0 loss)."
    )

    # 3. Instantiate MetisAgent
    agent = MetisAgent(
        n_var=65,
        n_obj=2,
        bounds=[(0.0, 1.0)] * 65,
        population_size=pop_size,
        initial_population=initial_pop_arr,
        max_elites=100,
        problem_context=problem_context
    )
    # Reference point for Hypervolume (Rx = 5.0e8 parameters, Ry = 6.0 validation loss)
    agent.ref = [500000000.0, 6.0]
    
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
        
        # Save generated sampling code
        code_file = f"checkpoints/agentic-optim/generation_{gen+1}_code.py"
        with open(code_file, 'w') as f:
            f.write(agent.last_code)
        print(f"✓ Saved generated sampling code: {code_file}")

        # Batch evaluation
        F = []
        state_dicts = []
        for idx in range(pop_size):
            x = X[idx]
            nodes, edges = vector_to_multilayer_graph(x, vocab_size, block_size, d_model=d_model)
            
            # Calculate parameter count
            dummy_model = RWNNGraph(nodes, edges, global_d_model=d_model)
            params = sum(p.numel() for p in dummy_model.parameters())
            
            # Distance-Based Ancestry Matching for Lamarckian Weight Inheritance / Resume Training
            parent_state = None
            if len(agent.pf_X) > 0:
                Xp_arr = np.array(agent.pf_X)
                dists = np.linalg.norm(Xp_arr - x, axis=1)
                best_idx = np.argmin(dists)
                min_dist = dists[best_idx]
                if min_dist < 1e-5:  # Exact match for promoted elites!
                    print(f" -> Promoted elite exact match. Resuming from its previous checkpoint.")
                    p_best = agent.pf_X[best_idx]
                    parent_state = X_hash_to_state.get(tuple(p_best))
                elif min_dist < 0.6:  # Candidate is in the evolutionary neighborhood of parent
                    p_best = agent.pf_X[best_idx]
                    parent_state = X_hash_to_state.get(tuple(p_best))
            
            # Train and evaluate on GPU
            print(f"Evaluating candidate {idx+1}/{pop_size} (Params: {params:,})...")
            try:
                # Proactive GPU memory pre-cleanup
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                trained_model, val_loss = train_and_eval_bpe_model(
                    nodes, edges, d_model=d_model, max_iters=eval_steps, batch_size=8, block_size=block_size,
                    parent_state_dict=parent_state # Inherit parent weights!
                )
                if val_loss >= 6.0:
                    # Enforce strict validation loss constraint < 6.0
                    val_loss = 99.9
                    s_dict = None
                else:
                    s_dict = trained_model.state_dict()
                
                # Delete model references to free up memory
                del trained_model
                del dummy_model
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Evaluation failed: {e}")
                val_loss = 99.9
                s_dict = None
                
            F.append([val_loss, params])
            state_dicts.append(s_dict)
            
        F_arr = np.array(F)
        
        # Cache newly evaluated state-dicts
        for k in range(pop_size):
            if state_dicts[k] is not None:
                X_hash_to_state[tuple(X[k])] = state_dicts[k]
                
        # Dynamically compute hypervolume reference point from all current and historical candidates
        combined_F = []
        if len(all_historical_pareto_F) > 0:
            combined_F.extend(all_historical_pareto_F)
        combined_F.extend([f for f in F_arr if f[0] < 6.0])
        
        if len(combined_F) > 0:
            combined_F_arr = np.array(combined_F)
            max_loss = np.max(combined_F_arr[:, 0])
            max_params = np.max(combined_F_arr[:, 1])
            # Set dynamic reference point with a 10% safety margin to strictly dominate all candidates
            agent.ref = [1.1 * max_loss, 1.1 * max_params]
            print(f" -> Dynamic Reference Point for HV calculation set to: Loss={agent.ref[0]:.4f}, Params={int(agent.ref[1]):,}")
            
        # Report results back to MetisAgent
        agent.tell(X, F_arr)

        # Get Pareto Front elites (restricted to loss < 6.0 and parents < 6.0)
        Xp, Fp = agent.result()
        
        # We manually filter elites that have loss < 6.0 as per constraint!
        valid_idx = [i for i, f in enumerate(Fp) if f[0] < 6.0]
        Xp = Xp[valid_idx] if len(valid_idx) > 0 else Xp
        Fp = Fp[valid_idx] if len(valid_idx) > 0 else Fp

        # Update historical Pareto front members list
        all_historical_pareto_F = [f for f in Fp]

        # Prune X_hash_to_state to keep only active Pareto front members' state dicts (Prevents memory bloat!)
        active_keys = set(tuple(x) for x in Xp)
        X_hash_to_state = {k: v for k, v in X_hash_to_state.items() if k in active_keys}

        print(f"\n🏆 Generation {gen + 1} Agentic Pareto Front:")
        
        pareto_reports = []
        for i, (x_elite, f_elite) in enumerate(zip(Xp, Fp)):
            loss_val = f_elite[0]
            param_val = int(f_elite[1])
            print(f"  Elite {i+1}: Params={param_val:,}, Loss={loss_val:.4f}")
            
            # Decode elite graph structure
            elite_nodes, elite_edges = vector_to_multilayer_graph(x_elite, vocab_size, block_size, d_model=d_model)
            
            # Save PyTorch weights (.pt) of the elite if present
            weight_file = "checkpoints/agentic-optim/no_weights.pt"
            for k in range(pop_size):
                if np.array_equal(X[k], x_elite) and state_dicts[k] is not None:
                    weight_file = f"checkpoints/agentic-optim/pareto_gen{gen+1}_ind{i+1}_loss{loss_val:.2f}.pt"
                    try:
                        torch.save(state_dicts[k], weight_file)
                    except Exception as save_err:
                        print(f" -> Warning: Skipped saving weights to disk ({save_err}). Continuing with in-memory caching.")
                    break
                    
            config_file = f"checkpoints/agentic-optim/pareto_gen{gen+1}_ind{i+1}_config.json"
            config_data = {
                'nodes': elite_nodes,
                'edges': elite_edges,
                'params': param_val,
                'loss': loss_val,
                'vector': x_elite.tolist()
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
                
            pareto_reports.append({
                'rank': i + 1,
                'nodes': len(elite_nodes),
                'edges': len(elite_edges),
                'params': param_val,
                'loss': loss_val,
                'saved_weights': weight_file,
                'saved_config': config_file
            })

        # Save Generation Report JSON
        report_file = f"checkpoints/agentic-optim/generation_{gen+1}_report.json"
        with open(report_file, 'w') as f:
            json.dump(pareto_reports, f, indent=4)
        print(f"✓ Saved {report_file}")

        # Plot and save Pareto Front PNG
        plt.figure(figsize=(8, 6))
        valid_F = F_arr[F_arr[:, 0] < 6.0]
        if len(valid_F) > 0:
            all_x = valid_F[:, 1]
            all_y = valid_F[:, 0]
            plt.scatter(all_x, all_y, color='#555555', alpha=0.6, label='Evaluated Population (< 6.0)')
            
        elite_valid = Fp[Fp[:, 0] < 6.0]
        if len(elite_valid) > 0:
            elite_x = elite_valid[:, 1]
            elite_y = elite_valid[:, 0]
            plt.scatter(elite_x, elite_y, color='#ff3333', s=100, marker='*', label='Pareto Frontier (Elites)')
            
            elites_sorted_idx = np.argsort(elite_x)
            if len(elite_x) > 1:
                plt.plot(elite_x[elites_sorted_idx], elite_y[elites_sorted_idx], color='#ff3333', linestyle='--', alpha=0.8)
            
        plt.xlabel("Complexity (Trainable Parameter Count)")
        plt.ylabel("Validation Loss (Cross-Entropy)")
        plt.title(f"Gen {gen+1} Agentic Optimization Pareto Front (< 6.0 Loss)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
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
    run_agentic_optimization(generations=100, pop_size=10, eval_steps=57860)
