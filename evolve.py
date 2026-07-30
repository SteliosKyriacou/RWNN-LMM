import random
import torch
from rwnn.graph import RWNNGraph
from rwnn.mutator import GraphMutator, get_canonical_nano_gpt
from train import train_rwnn_llm, CharTokenizer, DATA_FILE, download_data

# Simple Pareto Frontier extraction helper
def get_pareto_front(population):
    """
    Identifies the non-dominated individuals in a population.
    population: list of dicts with keys 'nodes', 'edges', 'loss', 'params'
    Returns: list of non-dominated individuals
    """
    pareto_front = []
    for i, ind in enumerate(population):
        dominated = False
        for j, other in enumerate(population):
            if i == j:
                continue
            # other dominates ind if it is better in at least one and not worse in both
            # Better = lower loss, lower params
            cond_loss = other['loss'] <= ind['loss']
            cond_params = other['params'] <= ind['params']
            cond_strict = (other['loss'] < ind['loss']) or (other['params'] < ind['params'])
            if cond_loss and cond_params and cond_strict:
                dominated = True
                break
        if not dominated:
            pareto_front.append(ind)
    return pareto_front


def run_evolutionary_loop(generations=3, pop_size=6, eval_steps=40):
    print("=== Phase 3: Multi-Objective Memetic Optimization Loop ===")
    
    # 1. Initialize tokenizer and context block size
    download_data()
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    block_size = 64
    d_model = 128

    mutator = GraphMutator(vocab_size, block_size, d_model)

    # 2. Setup initial population
    print(f"Initializing population of size {pop_size}...")
    population = []
    
    # Base canonical nanoGPT seed
    base_nodes, base_edges = get_canonical_nano_gpt(vocab_size, block_size, d_model)
    
    for idx in range(pop_size):
        if idx == 0:
            # Seed 1: perfect canonical nanoGPT
            nodes, edges = base_nodes, base_edges
            print(f"Ind {idx}: Initialized with canonical nanoGPT seed.")
        elif idx == 1:
            # Seed 2: slightly mutated nanoGPT
            nodes, edges = mutator.mutate(base_nodes, base_edges)
            print(f"Ind {idx}: Initialized with mutated nanoGPT.")
        else:
            # Seed 3+: heavily mutated variants to explore complexity frontier
            nodes, edges = base_nodes, base_edges
            for _ in range(3):
                nodes, edges = mutator.mutate(nodes, edges)
            print(f"Ind {idx}: Initialized with heavily mutated nanoGPT.")

        # Compile and calculate parameter count
        model = RWNNGraph(nodes, edges, global_d_model=d_model)
        params = sum(p.numel() for p in model.parameters())

        population.append({
            'nodes': nodes,
            'edges': edges,
            'params': params,
            'loss': float('inf') # Will be evaluated
        })

    # 3. Evolutionary Search Generational Loop
    for gen in range(generations):
        print(f"\n--- Generation {gen + 1} / {generations} ---")
        
        # Evaluate unevaluated individuals (Memetic optimization step: backpropagation)
        for idx, ind in enumerate(population):
            if ind['loss'] == float('inf'):
                print(f"Evaluating Individual {idx + 1}/{pop_size} (running {eval_steps} backprop steps)...")
                try:
                    _, val_loss = train_rwnn_llm(
                        nodes=ind['nodes'],
                        edges=ind['edges'],
                        max_iters=eval_steps,
                        batch_size=16,
                        block_size=block_size,
                        lr=1e-3
                    )
                    ind['loss'] = val_loss
                except Exception as e:
                    print(f"Evaluation failed for Ind {idx + 1}: {e}")
                    ind['loss'] = 99.9 # Penalty loss for bad graphs
                print(f"-> Result: Params = {ind['params']:,}, Val Loss = {ind['loss']:.4f}")

        # Extract and print Pareto Front
        pareto_front = get_pareto_front(population)
        print(f"\n🏆 Current Pareto Front (Non-Dominated Solutions):")
        for i, elite in enumerate(pareto_front):
            print(f"  Elite {i+1}: Nodes={len(elite['nodes'])}, Edges={len(elite['edges'])}, Params={elite['params']:,}, Loss={elite['loss']:.4f}")

        # If it's the last generation, we don't need to produce offspring
        if gen == generations - 1:
            break

        # Reproduce & Breed Next Generation
        print("\nBreeding next generation...")
        next_population = list(pareto_front) # Preserve elites (elitism)

        # Breed until next population is full
        while len(next_population) < pop_size:
            # Selection (random tournament from current population)
            parent_a = random.choice(population)
            parent_b = random.choice(population)

            # Crossover
            child_nodes, child_edges = mutator.crossover(
                (parent_a['nodes'], parent_a['edges']),
                (parent_b['nodes'], parent_b['edges'])
            )

            # Mutation
            child_nodes, child_edges = mutator.mutate(child_nodes, child_edges)

            # Check if offspring has valid shape and compile
            try:
                model = RWNNGraph(child_nodes, child_edges, global_d_model=d_model)
                params = sum(p.numel() for p in model.parameters())
                next_population.append({
                    'nodes': child_nodes,
                    'edges': child_edges,
                    'params': params,
                    'loss': float('inf') # To be evaluated in next generation
                })
            except Exception:
                continue # Retry if offspring compiles with errors

        population = next_population[:pop_size]

    print("\n=== Evolutionary Search Complete! ===")
    print("Final Pareto Front Solutions:")
    for i, elite in enumerate(pareto_front):
        print(f"Solution {i+1} [Nodes: {len(elite['nodes'])}, Edges: {len(elite['edges'])}]: Params = {elite['params']:,}, Val Loss = {elite['loss']:.4f}")


if __name__ == "__main__":
    run_evolutionary_loop(generations=2, pop_size=4, eval_steps=20)
