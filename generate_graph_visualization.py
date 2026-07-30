import os
import networkx as nx
import matplotlib.pyplot as plt
from rwnn.mutator import get_canonical_nano_gpt

def generate_visualizations():
    print("=== Generating Graph Visualizations using NetworkX ===")
    
    # 1. Create Output Folder for Assets if needed
    os.makedirs("assets", exist_ok=True)

    # 2. Setup colors for different node types
    color_map = {
        'input': '#ff99ff',       # Pink
        'token_embedding': '#99ccff', # Light Blue
        'positional_embedding': '#99ccff',
        'layer_norm': '#99ff99',  # Light Green
        'causal_attention': '#ffcc99', # Light Orange
        'sum': '#ffff99',         # Light Yellow
        'linear': '#ffb3b3',      # Soft Red
        'activation': '#e6ccff'   # Soft Purple
    }

    # ==================== PLOT 1: Canonical nanoGPT DAG ====================
    print("Generating nanoGPT DAG visualization...")
    nodes, edges = get_canonical_nano_gpt()
    
    G = nx.DiGraph()
    for n in nodes:
        # We simplify labels for beauty
        label = f"{n['id']}: {n['type']}"
        G.add_node(n['id'], label=label, type=n['type'])
    G.add_edges_from(edges)

    plt.figure(figsize=(12, 10))
    
    # Use a structured layout (layered multipartite layout by topological generation or hierarchical pos)
    # Let's manually define a beautiful top-to-bottom layout
    pos = {
        0: (0, 10),      # Input
        1: (-2, 8.5),    # WTE
        2: (2, 8.5),     # WPE
        3: (0, 7),       # Embed Sum
        4: (0, 5.5),     # LN 1
        5: (2, 4),       # Attention
        6: (0, 2.5),     # Attn Sum
        7: (-2, 1),      # LN 2
        8: (-2, -0.5),   # MLP Up
        9: (-2, -2),     # GELU
        10: (-2, -3.5),  # MLP Down
        11: (0, -5),     # MLP Sum
        12: (0, -6.5),   # LN Final
        13: (0, -8)      # Head
    }

    node_colors = [color_map.get(G.nodes[n]['type'], '#cccccc') for n in G.nodes]
    labels = nx.get_node_attributes(G, 'label')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2800, edgecolors='#333333', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='#666666', width=1.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif', font_weight='bold')

    plt.title("Canonical Single-Layer nanoGPT Compiled as a H-DAG", fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("assets/nanogpt_dag_graph.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved assets/nanogpt_dag_graph.png")

    # ==================== PLOT 2: Atomic LayerNorm Sub-Graph ====================
    print("Generating Atomic LayerNorm Sub-Graph visualization...")
    # Node mapping for our atomic LayerNorm
    atomic_nodes = [
        {'id': 0, 'type': 'input'},
        {'id': 1, 'type': 'mean_reduce'},
        {'id': 2, 'type': 'subtract'},
        {'id': 3, 'type': 'square'},
        {'id': 4, 'type': 'mean_reduce'},
        {'id': 5, 'type': 'sqrt'},
        {'id': 6, 'type': 'divide'},
        {'id': 7, 'type': 'scale_shift'}
    ]
    
    atomic_edges = [
        (0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (5, 6), (6, 7)
    ]

    G_atomic = nx.DiGraph()
    for n in atomic_nodes:
        G_atomic.add_node(n['id'], label=f"{n['id']}: {n['type']}", type=n['type'])
    G_atomic.add_edges_from(atomic_edges)

    # Let's manually layout the atomic nodes beautifully
    pos_atomic = {
        0: (0, 8),     # Input x
        1: (-2, 6.5),  # Mean Reduce
        2: (0, 5),     # Subtract (x - Mean)
        3: (2, 3.5),   # Square
        4: (2, 2),     # Mean Reduce (Variance)
        5: (2, 0.5),   # Sqrt (Std Dev)
        6: (0, -1),    # Divide
        7: (0, -2.5)   # Scale Shift
    }

    # Setup colors for atomic operations
    atomic_colors = {
        'input': '#ff99ff',
        'mean_reduce': '#b3f0ff',
        'subtract': '#ffffb3',
        'square': '#ffd1b3',
        'sqrt': '#ffb3d1',
        'divide': '#ff9999',
        'scale_shift': '#b3ffb3'
    }

    node_colors_atomic = [atomic_colors.get(G_atomic.nodes[n]['type'], '#cccccc') for n in G_atomic.nodes]
    labels_atomic = nx.get_node_attributes(G_atomic, 'label')

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G_atomic, pos_atomic, node_color=node_colors_atomic, node_size=3200, edgecolors='#333333', linewidths=1.5)
    nx.draw_networkx_edges(G_atomic, pos_atomic, arrowstyle='->', arrowsize=22, edge_color='#555555', width=1.5)
    nx.draw_networkx_labels(G_atomic, pos_atomic, labels_atomic, font_size=9, font_family='sans-serif', font_weight='bold')

    plt.title("LayerNorm Decomposed into a Sub-Graph of Atomic Nodes", fontsize=13, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("assets/layernorm_atomic_graph.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved assets/layernorm_atomic_graph.png")

if __name__ == "__main__":
    generate_visualizations()
