import os
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rwnn.graph import RWNNGraph
from rwnn.mutator import GraphMutator, get_gpt2_dag, is_valid_dag
from train import CharTokenizer, DATA_FILE, download_data

# Simple Pareto Frontier extraction helper
def get_pareto_front(population):
    """
    Identifies the non-dominated individuals in a population.
    population: list of dicts with keys 'nodes', 'edges', 'loss', 'params'
    """
    pareto_front = []
    for i, ind in enumerate(population):
        dominated = False
        for j, other in enumerate(population):
            if i == j:
                continue
            cond_loss = other['loss'] <= ind['loss']
            cond_params = other['params'] <= ind['params']
            cond_strict = (other['loss'] < ind['loss']) or (other['params'] < ind['params'])
            if cond_loss and cond_params and cond_strict:
                dominated = True
                break
        if not dominated:
            pareto_front.append(ind)
    return pareto_front


def train_and_eval_bpe_model(nodes, edges, d_model, max_iters=100, batch_size=16, block_size=128):
    """Trains a compiled H-DAG model on BPE tokens and returns validation loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load compiled BPE datasets
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    vocab_size = 50257

    model = RWNNGraph(nodes, edges, global_d_model=d_model)
    model.to(device)

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((d[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((d[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    # 2. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # 3. Fast Train
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

    # 4. Evaluate Val Loss
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


def run_background_evolution(generations=3, pop_size=4, eval_steps=100):
    print("=== BACKGROUND EVOLUTION KICKOFF ===")
    os.makedirs("checkpoints", exist_ok=True)
    
    vocab_size = 50257
    block_size = 256
    d_model = 128 # Kept light to prevent memory leak and make background process very fast

    mutator = GraphMutator(vocab_size, block_size, d_model)

    # 1. Initialize population of GPT2 and nanoGPT variants
    population = []
    
    # Core Seeds
    toy_nodes, toy_edges = get_gpt2_dag('toy', vocab_size, block_size)
    gpt2_nodes, gpt2_edges = get_gpt2_dag('gpt2', vocab_size, block_size)

    for idx in range(pop_size):
        if idx == 0:
            # Seed 1: nanoGPT Toy
            nodes, edges, model_name = toy_nodes, toy_edges, 'nanoGPT_toy'
        elif idx == 1:
            # Seed 2: GPT-2
            nodes, edges, model_name = gpt2_nodes, gpt2_edges, 'gpt2_124m'
        elif idx == 2:
            # Seed 3: Mutated nanoGPT
            nodes, edges = mutator.mutate(toy_nodes, toy_edges)
            model_name = 'mutated_nanoGPT_toy'
        else:
            # Seed 4: Mutated GPT-2
            nodes, edges = mutator.mutate(gpt2_nodes, gpt2_edges)
            model_name = 'mutated_gpt2'

        # Get parameter count
        dummy_model = RWNNGraph(nodes, edges, global_d_model=d_model)
        params = sum(p.numel() for p in dummy_model.parameters())

        population.append({
            'nodes': nodes,
            'edges': edges,
            'params': params,
            'loss': float('inf'),
            'type': model_name,
            'gen_born': 0
        })

    # 2. Generational Loop
    for gen in range(generations):
        print(f"\n--- Generation {gen + 1} / {generations} ---")
        
        # Evaluate individuals
        for idx, ind in enumerate(population):
            if ind['loss'] == float('inf'):
                print(f"Evaluating individual {idx+1}/{pop_size}...")
                try:
                    trained_model, val_loss = train_and_eval_bpe_model(
                        ind['nodes'], ind['edges'], d_model=d_model, max_iters=eval_steps, block_size=64
                    )
                    ind['loss'] = val_loss
                    ind['state_dict'] = trained_model.state_dict() # Cache temporarily to save
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    ind['loss'] = 99.9

        # Identify current Pareto Front
        pareto_front = get_pareto_front(population)
        print(f"\n🏆 Generation {gen + 1} Pareto Front:")
        
        # Save Pareto front configurations and PyTorch state-dicts
        pareto_reports = []
        for i, elite in enumerate(pareto_front):
            print(f"  Elite {i+1}: Nodes={len(elite['nodes'])}, Edges={len(elite['edges'])}, Params={elite['params']:,}, Loss={elite['loss']:.4f}")
            
            # Save PyTorch weight file (.pt) for the Pareto front model
            # We save it safely inside checkpoints directory
            weight_file = f"checkpoints/pareto_gen{gen+1}_ind{i+1}_loss{elite['loss']:.2f}.pt"
            if 'state_dict' in elite:
                torch.save(elite['state_dict'], weight_file)
                
            # Create json configuration so we can completely rebuild the H-DAG
            config_file = f"checkpoints/pareto_gen{gen+1}_ind{i+1}_config.json"
            config_data = {
                'nodes': elite['nodes'],
                'edges': elite['edges'],
                'params': elite['params'],
                'loss': elite['loss'],
                'type': elite['type']
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
                
            pareto_reports.append({
                'rank': i + 1,
                'nodes': len(elite['nodes']),
                'edges': len(elite['edges']),
                'params': elite['params'],
                'loss': elite['loss'],
                'type': elite['type'],
                'saved_weights': weight_file,
                'saved_config': config_file
            })

        # Save Generation Report JSON
        report_file = f"checkpoints/generation_{gen+1}_report.json"
        with open(report_file, 'w') as f:
            json.dump(pareto_reports, f, indent=4)
        print(f"✓ Saved {report_file}")

        # Plot and save Pareto Front PNG visualization
        plt.figure(figsize=(8, 6))
        all_x = [ind['params'] for ind in population if ind['loss'] < 90.0]
        all_y = [ind['loss'] for ind in population if ind['loss'] < 90.0]
        elite_x = [elite['params'] for elite in pareto_front if elite['loss'] < 90.0]
        elite_y = [elite['loss'] for elite in pareto_front if elite['loss'] < 90.0]
        
        plt.scatter(all_x, all_y, color='#555555', alpha=0.6, label='Evaluated Population')
        plt.scatter(elite_x, elite_y, color='#ff3333', s=100, marker='*', label='Pareto Frontier (Elites)')
        
        # Draw Pareto frontier line (sorted by params)
        elites_sorted = sorted(pareto_front, key=lambda x: x['params'])
        elites_sorted = [e for e in elites_sorted if e['loss'] < 90.0]
        if len(elites_sorted) > 1:
            plt.plot([e['params'] for e in elites_sorted], [e['loss'] for e in elites_sorted], color='#ff3333', linestyle='--', alpha=0.8)
            
        plt.xlabel("Complexity (Trainable Parameter Count)")
        plt.ylabel("Validation Loss (Cross-Entropy)")
        plt.title(f"Gen {gen+1} Multi-Objective Architectural Pareto Front")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        plot_file = f"checkpoints/generation_{gen+1}_pareto.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Pareto plot: {plot_file}")

        # Remove state_dicts from current population list to avoid memory buildup
        for ind in population:
            if 'state_dict' in ind:
                del ind['state_dict']

        if gen == generations - 1:
            break

        # Reproduce & Breed Next Gen
        print("\nBreeding next generation...")
        next_population = list(pareto_front) # Elitist preservation
        
        while len(next_population) < pop_size:
            parent_a = random.choice(population)
            parent_b = random.choice(population)
            
            child_nodes, child_edges = mutator.crossover(
                (parent_a['nodes'], parent_a['edges']),
                (parent_b['nodes'], parent_b['edges'])
            )
            child_nodes, child_edges = mutator.mutate(child_nodes, child_edges)
            
            try:
                dummy_model = RWNNGraph(child_nodes, child_edges, global_d_model=d_model)
                params = sum(p.numel() for p in dummy_model.parameters())
                next_population.append({
                    'nodes': child_nodes,
                    'edges': child_edges,
                    'params': params,
                    'loss': float('inf'),
                    'type': f"hybrid_gen{gen+1}",
                    'gen_born': gen + 1
                })
            except Exception:
                continue

        population = next_population[:pop_size]

    print("\n=== BACKGROUND EVOLUTION LOOP COMPLETE ===")


if __name__ == "__main__":
    run_background_evolution(generations=3, pop_size=10, eval_steps=1000)
