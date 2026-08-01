import os
import sys
import json
import time
import math
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

def graph_to_vector(nodes, edges):
    """Encodes a discrete graph configuration back into a continuous vector x of shape [86]."""
    x = np.zeros(86)
    
    # 1. Encode node types
    pool = ['layer_norm', 'linear', 'causal_attention', 'activation', 'sum', 'concat', 'dropout']
    for i in range(8):
        node_id = 4 + i
        # Find node type in nodes
        ntype = 'layer_norm' # default
        for n in nodes:
            if n['id'] == node_id:
                ntype = n['type']
                break
        if ntype in pool:
            pool_idx = pool.index(ntype)
            x[i] = (pool_idx + 0.5) / len(pool)
            
    # 2. Encode edges (upper triangular directed adjacency matrix)
    edge_idx = 0
    edges_set = set(tuple(e) for e in edges)
    for u in range(13):
        for v in range(u + 1, 13):
            if (u, v) in edges_set:
                x[8 + edge_idx] = 1.0
            else:
                x[8 + edge_idx] = 0.0
            edge_idx += 1
            
    return x


def vector_to_graph(x, vocab_size=50257, block_size=256, d_model=192):
    """Decodes a continuous vector x of shape [86] back into a discrete graph configuration."""
    nodes_vars = x[:8]
    edges_vars = x[8:]
    
    # 1. Rebuild nodes
    nodes = [
        {'id': 0, 'type': 'input', 'kwargs': {}},
        {'id': 1, 'type': 'token_embedding', 'kwargs': {'vocab_size': vocab_size, 'd_model': d_model}},
        {'id': 2, 'type': 'positional_embedding', 'kwargs': {'max_seq_len': block_size, 'd_model': d_model}},
        {'id': 3, 'type': 'sum', 'kwargs': {}}
    ]
    
    pool = ['layer_norm', 'linear', 'causal_attention', 'activation', 'sum', 'concat', 'dropout']
    for i, v in enumerate(nodes_vars):
        node_id = 4 + i
        pool_idx = int(v * len(pool))
        pool_idx = min(pool_idx, len(pool) - 1)
        ntype = pool[pool_idx]
        
        # Configure kwargs
        kwargs = {}
        if ntype == 'layer_norm':
            kwargs = {'d_model': d_model}
        elif ntype == 'linear':
            kwargs = {'d_in': d_model, 'd_out': d_model}
        elif ntype == 'causal_attention':
            kwargs = {'n_head': 12, 'd_model': d_model, 'dropout': 0.1} # Division by 12 works with d_model=192!
        elif ntype == 'activation':
            kwargs = {'act_type': 'gelu'}
        elif ntype == 'concat':
            kwargs = {'dim': -1}
        elif ntype == 'dropout':
            kwargs = {'dropout': 0.1}
            
        nodes.append({'id': node_id, 'type': ntype, 'kwargs': kwargs})
        
    # Output head (always node 12)
    nodes.append({'id': 12, 'type': 'linear', 'kwargs': {'d_in': d_model, 'd_out': vocab_size}})
    
    # 2. Rebuild edges
    edges = []
    edge_idx = 0
    for u in range(13):
        for v in range(u + 1, 13):
            val = edges_vars[edge_idx]
            edge_idx += 1
            if val > 0.5:
                # Apply input boundaries
                if u == 0 and v not in {1, 2}:
                    continue
                if v in {1, 2} and u != 0:
                    continue
                edges.append((u, v))
                
    # 3. Validation: if cyclic or disconnected, fallback to toy config
    if is_valid_dag(nodes, edges):
        return nodes, edges
    return get_gpt2_dag('toy', vocab_size, block_size, override_d_model=d_model)


def train_and_eval_bpe_model(nodes, edges, d_model=192, max_iters=1000, batch_size=16, block_size=128):
    """Trains a compiled H-DAG model on BPE tokens and returns validation loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load compiled BPE datasets
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')

    model = RWNNGraph(nodes, edges, global_d_model=d_model)
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
    os.makedirs("checkpoints/agentic-optim", exist_ok=True)
    
    vocab_size = 50257
    block_size = 256
    d_model = 192

    # 1. Seed Gen 0 with the best Gen 99 elites if available
    initial_population = []
    previous_seeds = [
        "checkpoints/vanila evolutionary algorithm/pareto_gen99_ind1_config.json",
        "checkpoints/vanila evolutionary algorithm/pareto_gen99_ind3_config.json"
    ]
    loaded_seeds_count = 0
    for seed_path in previous_seeds:
        if os.path.exists(seed_path):
            try:
                with open(seed_path, 'r') as f:
                    seed_data = json.load(f)
                x_vec = graph_to_vector(seed_data['nodes'], seed_data['edges'])
                initial_population.append(x_vec)
                loaded_seeds_count += 1
            except Exception as e:
                print(f"Failed to load previous seed {seed_path}: {e}")
                
    print(f"Loaded {loaded_seeds_count} previous optimal elites to seed Gen 0.")
    
    # Fill up the rest of the Gen 0 population with random vectors
    while len(initial_population) < pop_size:
        initial_population.append(np.random.uniform(0.0, 1.0, 86))
        
    initial_pop_arr = np.array(initial_population)

    # 2. Plain-Text Auxiliary Context (Domain Knowledge Injection)
    problem_context = (
        "You are optimizing a Randomly Wired Large Language Model (RWNN-LLM) represented as "
        "a Heterogeneous Directed Acyclic Graph (H-DAG) of modular tensor operations.\n\n"
        "## Decision Variable Guide:\n"
        "- Variables 0 to 7 represent the modular operator types of 8 hidden graph nodes. The range [0, 1] maps sequentially to: ['layer_norm', 'linear', 'causal_attention', 'activation', 'sum', 'concat', 'dropout'].\n"
        "- Variables 8 to 85 represent the upper-triangular directed connectivity matrix between the 13 nodes of the network. If a variable > 0.5, the directed edge (u -> v) exists.\n\n"
        "## Domain Knowledge & Structural Physics:\n"
        "1. Symmetrical Flow: Nodes with smaller IDs are closer to the embedding inputs, while nodes with larger IDs are closer to the final vocabulary output head.\n"
        "2. Residual Channels: Standard Transformers rely on deep residual streams. Connecting a low-level node directly to a high-level sum node creates an active residual bypass, preventing gradient explosion/vanishing.\n"
        "3. Node Synergy: Attention layers must always be preceded by normalization (LayerNorm) to prevent high-dimensional variance drift, and should be followed by a projection and non-linear activation.\n"
        "4. Channel Bottlenecks: Connecting multiple nodes side-by-side to a 'concat' node aggregates their channels, widening representation width, while 'linear' and 'matmul' contract them.\n"
        "5. Sparsity: If you set connection variables to <= 0.5, you delete those edges. Leaving a node completely disconnected effectively prunes it (it outputs zeros), reducing complexity and parameter count.\n\n"
        "## Objectives and Constraints:\n"
        "- Minimizing Objective 1: Validation Cross-Entropy Loss (Perplexity). You must keep loss strictly < 5.0. Any loss >= 5.0 is a complete failure.\n"
        "- Minimizing Objective 2: Trainable parameter count (Complexity).\n"
        "- Maintain a stable trade-off frontier. Deeper networks (with active residual bypasses) will have more parameters but achieve lower loss."
    )

    # 3. Instantiate MetisAgent
    agent = MetisAgent(
        n_var=86,
        n_obj=2,
        bounds=[(0.0, 1.0)] * 86,
        population_size=pop_size,
        initial_population=initial_pop_arr,
        max_elites=100,
        problem_context=problem_context
    )
    # Reference point for Hypervolume (Rx = 2.5e7 parameters, Ry = 5.0 validation loss)
    agent.ref = [2.5e7, 5.0]

    # 4. Agentic Optimization Loop
    for gen in range(generations):
        print(f"\n--- Agentic Generation {gen + 1} / {generations} ---")
        
        # Agent writes Python code and proposes candidates
        X = agent.ask()
        
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
            nodes, edges = vector_to_graph(x, vocab_size, block_size, d_model=d_model)
            
            # Calculate parameter count
            dummy_model = RWNNGraph(nodes, edges, global_d_model=d_model)
            params = sum(p.numel() for p in dummy_model.parameters())
            
            # Train and evaluate on GPU
            print(f"Evaluating candidate {idx+1}/{pop_size} (Params: {params:,})...")
            try:
                trained_model, val_loss = train_and_eval_bpe_model(
                    nodes, edges, d_model=d_model, max_iters=eval_steps, block_size=64
                )
                if val_loss >= 5.0:
                    # Enforce strict validation loss constraint < 5.0
                    val_loss = 99.9
                    s_dict = None
                else:
                    s_dict = trained_model.state_dict()
            except Exception as e:
                print(f"Evaluation failed: {e}")
                val_loss = 99.9
                s_dict = None
                
            F.append([val_loss, params])
            state_dicts.append(s_dict)
            
        F_arr = np.array(F)
        
        # Report results back to MetisAgent
        agent.tell(X, F_arr)

        # Get Pareto Front elites
        Xp, Fp = agent.result()
        print(f"\n🏆 Generation {gen + 1} Agentic Pareto Front:")
        
        pareto_reports = []
        for i, (x_elite, f_elite) in enumerate(zip(Xp, Fp)):
            loss_val = f_elite[0]
            param_val = int(f_elite[1])
            print(f"  Elite {i+1}: Params={param_val:,}, Loss={loss_val:.4f}")
            
            # Decode elite graph structure
            elite_nodes, elite_edges = vector_to_graph(x_elite, vocab_size, block_size, d_model=d_model)
            
            # Save PyTorch weights (.pt) of the elite if present
            # We match using decision vector exact index matching
            weight_file = "checkpoints/agentic-optim/no_weights.pt"
            for k in range(pop_size):
                if np.array_equal(X[k], x_elite) and state_dicts[k] is not None:
                    weight_file = f"checkpoints/agentic-optim/pareto_gen{gen+1}_ind{i+1}_loss{loss_val:.2f}.pt"
                    torch.save(state_dicts[k], weight_file)
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
        # filter out failed ones
        valid_F = F_arr[F_arr[:, 0] < 90.0]
        all_x = valid_F[:, 1]
        all_y = valid_F[:, 0]
        elite_x = Fp[Fp[:, 0] < 90.0, 1]
        elite_y = Fp[Fp[:, 0] < 90.0, 0]
        
        plt.scatter(all_x, all_y, color='#555555', alpha=0.6, label='Evaluated Population')
        plt.scatter(elite_x, elite_y, color='#ff3333', s=100, marker='*', label='Pareto Frontier (Elites)')
        
        # Sort and plot elites line
        elites_sorted_idx = np.argsort(elite_x)
        if len(elite_x) > 1:
            plt.plot(elite_x[elites_sorted_idx], elite_y[elites_sorted_idx], color='#ff3333', linestyle='--', alpha=0.8)
            
        plt.xlabel("Complexity (Trainable Parameter Count)")
        plt.ylabel("Validation Loss (Cross-Entropy)")
        plt.title(f"Gen {gen+1} Agentic Optimization Pareto Front")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        plot_file = f"checkpoints/agentic-optim/generation_{gen+1}_pareto.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Pareto plot: {plot_file}")

    print("\n=== AGENTIC OPTIMIZATION LOOP COMPLETE ===")


if __name__ == "__main__":
    run_agentic_optimization(generations=100, pop_size=10, eval_steps=1000)
