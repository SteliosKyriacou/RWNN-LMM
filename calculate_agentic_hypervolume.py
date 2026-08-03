import os
import json
import numpy as np
import matplotlib.pyplot as plt

def calculate_agentic_hypervolume():
    print("=== Calculating Agentic Hypervolume Progression ===")
    
    # Reference point for normalization (matching our evolve_agentic.py ref point)
    Rx = 2.5e7  # 25 Million parameters reference
    Ry = 5.0    # 5.0 validation loss reference
    
    total_ref_area = Rx * Ry
    
    generations = []
    hypervolumes = []
    best_losses = []
    best_params = []
    
    # Scan completed reports on disk
    for gen in range(1, 101):
        report_file = f"checkpoints/agentic-optim/generation_{gen}_report.json"
        if not os.path.exists(report_file):
            continue
            
        with open(report_file, 'r') as f:
            elites = json.load(f)
            
        # Filter out elites that don't satisfy Rx and Ry boundaries (only keep valid sub-5.0 models)
        valid_elites = [e for e in elites if e['params'] < Rx and e['loss'] < Ry]
        if not valid_elites:
            continue
            
        # Sort elites by parameters ascending
        valid_elites = sorted(valid_elites, key=lambda x: x['params'])
        
        # Calculate mathematically exact dominated area
        x = [e['params'] for e in valid_elites]
        y = [e['loss'] for e in valid_elites]
        
        area = (Rx - x[0]) * (Ry - y[0])
        for j in range(1, len(x)):
            area += (Rx - x[j]) * (y[j-1] - y[j])
            
        # Normalize hypervolume
        hv = area / total_ref_area
        
        generations.append(gen)
        hypervolumes.append(hv)
        best_losses.append(min(y))
        best_params.append(min(x))

    n_gens = len(generations)
    print(f"Calculated hypervolume for {n_gens} completed generations.")
    print(f"Gen 1 Hypervolume: {hypervolumes[0]:.4f} (Best Loss: {best_losses[0]:.4f})")
    if n_gens >= 20:
        print(f"Gen 20 Hypervolume: {hypervolumes[19]:.4f} (Best Loss: {best_losses[19]:.4f})")
    if n_gens >= 45:
        print(f"Gen 45 Hypervolume: {hypervolumes[44]:.4f} (Best Loss: {best_losses[44]:.4f})")

    # 1. Plot Hypervolume progression
    plt.figure(figsize=(10, 5))
    plt.plot(generations, hypervolumes, color='#ff3333', linewidth=2.5, label='Agentic Pareto Hypervolume (S-Metric)')
    plt.fill_between(generations, hypervolumes, color='#ff3333', alpha=0.1)
    
    plt.xlabel("Generation Index", fontsize=11, fontweight='bold')
    plt.ylabel("Normalized Hypervolume Area", fontsize=11, fontweight='bold')
    plt.title("Agentic Graph Optimization Convergence (Hypervolume Progression)", fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(1, max(generations))
    plt.legend(loc='lower right')
    
    os.makedirs("assets", exist_ok=True)
    plot_file = "assets/agentic_hypervolume_progression.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved convergence progression plot: {plot_file}")

    # 2. Plot best loss progression
    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_losses, color='#3399ff', linewidth=2.5, label='Best Validation Perplexity Loss')
    plt.xlabel("Generation Index", fontsize=11, fontweight='bold')
    plt.ylabel("Validation Cross-Entropy Loss", fontsize=11, fontweight='bold')
    plt.title("Fittest Agentic Architecture Validation Loss Convergence", fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(1, max(generations))
    plt.legend(loc='upper right')
    
    plot_file_loss = "assets/agentic_loss_progression.png"
    plt.savefig(plot_file_loss, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved loss progression plot: {plot_file_loss}")

if __name__ == "__main__":
    calculate_agentic_hypervolume()
