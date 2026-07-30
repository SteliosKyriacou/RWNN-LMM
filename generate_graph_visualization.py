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
        label = f"{n['id']}: {n['type']}"
        G.add_node(n['id'], label=label, type=n['type'])
    G.add_edges_from(edges)

    plt.figure(figsize=(12, 10))
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

    # ==================== PLOT 3: Pure Atomic Block DAG Graph ====================
    print("Generating Complete Pure Atomic Block DAG Graph...")
    # This represents the nanoGPT block completely deconstructed into basic math operators
    # Let's map out its layout
    block_nodes = [
        {'id': 0, 'type': 'input'},
        {'id': 1, 'type': 'token_embedding'},
        {'id': 2, 'type': 'positional_embedding'},
        {'id': 3, 'type': 'sum'}, # Embeddings Sum (D=384)
        
        # LayerNorm 1 primitives (8 nodes)
        {'id': 4, 'type': 'mean_reduce'},
        {'id': 5, 'type': 'subtract'},
        {'id': 6, 'type': 'square'},
        {'id': 7, 'type': 'mean_reduce'},
        {'id': 8, 'type': 'sqrt'},
        {'id': 9, 'type': 'divide'},
        {'id': 10, 'type': 'scale_shift'},
        
        # Q, K, V Dense projections uncoupled to MatMul/AddBias (6 nodes)
        {'id': 11, 'type': 'matmul'}, # Q weight
        {'id': 12, 'type': 'add_bias'}, # Q bias
        {'id': 13, 'type': 'matmul'}, # K weight
        {'id': 14, 'type': 'add_bias'}, # K bias
        {'id': 15, 'type': 'matmul'}, # V weight
        {'id': 16, 'type': 'add_bias'}, # V bias
        
        # Multi-Head reshape/transposes (6 nodes)
        {'id': 17, 'type': 'reshape'},
        {'id': 18, 'type': 'transpose'},
        {'id': 19, 'type': 'reshape'},
        {'id': 20, 'type': 'transpose'},
        {'id': 21, 'type': 'reshape'},
        {'id': 22, 'type': 'transpose'},
        
        # Attention score multiplication and softmax (3 nodes)
        {'id': 23, 'type': 'causal_batch_matmul'},
        {'id': 24, 'type': 'activation'}, # Softmax
        {'id': 25, 'type': 'causal_batch_matmul'}, # Attn x V
        
        # Squeeze/Transpose/Projection (4 nodes)
        {'id': 26, 'type': 'transpose'},
        {'id': 27, 'type': 'reshape'},
        {'id': 28, 'type': 'matmul'}, # Out weight
        {'id': 29, 'type': 'add_bias'}, # Out bias
        
        # Residual Sum (Attention)
        {'id': 30, 'type': 'sum'},
        
        # MLP Block atomic components (5 nodes)
        {'id': 31, 'type': 'matmul'}, # MLP Up weight
        {'id': 32, 'type': 'add_bias'}, # MLP Up bias
        {'id': 33, 'type': 'activation'}, # GELU
        {'id': 34, 'type': 'matmul'}, # MLP Down weight
        {'id': 35, 'type': 'add_bias'}, # MLP Down bias
        
        # Residual Sum (MLP) and head
        {'id': 36, 'type': 'sum'},
        {'id': 37, 'type': 'layer_norm'}, # Coarse norm at output
        {'id': 38, 'type': 'linear'} # Output head
    ]

    block_edges = [
        # Embeddings
        (0, 1), (0, 2), (1, 3), (2, 3),
        
        # LayerNorm 1
        (3, 4), (3, 5), (4, 5), (5, 6), (6, 7), (7, 8), (5, 9), (8, 9), (9, 10),
        
        # Attention inputs
        (10, 11), (11, 12),
        (10, 13), (13, 14),
        (10, 15), (15, 16),
        
        # Reshapes/transposes
        (12, 17), (17, 18),
        (14, 19), (19, 20),
        (16, 21), (21, 22),
        
        # Attention scores
        (18, 23), (20, 23), (23, 24),
        (24, 25), (22, 25),
        
        # Squeezes / Output Proj
        (25, 26), (26, 27), (27, 28), (28, 29),
        
        # Attention Residual
        (3, 30), (29, 30),
        
        # MLP Block
        (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
        
        # MLP Residual
        (30, 36), (35, 36),
        (36, 37), (37, 38)
    ]

    G_block = nx.DiGraph()
    for n in block_nodes:
        G_block.add_node(n['id'], label=f"{n['id']}: {n['type']}", type=n['type'])
    G_block.add_edges_from(block_edges)

    # Let's set up a beautiful vertical hierarchy pos map
    pos_block = {
        0: (0, 15),      # Input tokens
        1: (-3, 13.5),   # WTE
        2: (3, 13.5),    # WPE
        3: (0, 12),      # Embed Sum
        
        # LN 1 primitives
        4: (-2.5, 10.5), # Mean Reduce
        5: (0, 9.5),     # Subtract
        6: (2.5, 8.5),   # Square
        7: (2.5, 7.2),   # Mean Reduce
        8: (2.5, 6.0),   # Sqrt
        9: (0, 5.0),     # Divide
        10: (0, 3.8),    # Scale Shift
        
        # Q, K, V
        11: (-5, 2),     # Q matmul
        12: (-5, 0.8),   # Q bias
        13: (0, 2),      # K matmul
        14: (0, 0.8),    # K bias
        15: (5, 2),      # V matmul
        16: (5, 0.8),    # V bias
        
        # Transpose/Reshapes
        17: (-5, -0.4),  # Q reshape
        18: (-5, -1.6),  # Q transpose
        19: (0, -0.4),   # K reshape
        20: (0, -1.6),   # K transpose
        21: (5, -0.4),   # V reshape
        22: (5, -1.6),   # V transpose
        
        # Attention scores
        23: (-2.5, -2.8),# Causal Batch MM
        24: (-2.5, -4.0),# Softmax
        25: (2.5, -4.0), # Attn @ V
        
        # Out projections
        26: (2.5, -5.2), # Transpose
        27: (2.5, -6.4), # Reshape
        28: (0, -7.5),   # Out matmul
        29: (0, -8.7),   # Out bias
        
        # Residual 1 Sum
        30: (0, -10.0),  # Attn Residual Sum
        
        # MLP Block
        31: (-4, -11.5), # MLP Up matmul
        32: (-4, -12.5), # MLP Up bias
        33: (-4, -13.5), # GELU
        34: (-4, -14.5), # MLP Down matmul
        35: (-4, -15.5), # MLP Down bias
        
        # Output layers
        36: (0, -17.0),  # MLP Residual Sum
        37: (0, -18.2),  # Final LN
        38: (0, -19.5)   # Head
    }

    # Node coloring
    unique_types = list(set(n['type'] for n in block_nodes))
    import matplotlib.colors as mcolors
    color_palette = [
        '#ff99ff', '#99ccff', '#99ff99', '#ffcc99', '#ffff99', '#ffb3b3', '#e6ccff',
        '#b3f0ff', '#ffffb3', '#ffd1b3', '#ffb3d1', '#ff9999', '#b3ffb3'
    ]
    block_colors = {utype: color_palette[i % len(color_palette)] for i, utype in enumerate(unique_types)}

    node_colors_block = [block_colors.get(G_block.nodes[n]['type'], '#cccccc') for n in G_block.nodes]
    labels_block = nx.get_node_attributes(G_block, 'label')

    plt.figure(figsize=(16, 20))
    nx.draw_networkx_nodes(G_block, pos_block, node_color=node_colors_block, node_size=1800, edgecolors='#333333', linewidths=1.0)
    nx.draw_networkx_edges(G_block, pos_block, arrowstyle='->', arrowsize=15, edge_color='#777777', width=1.0)
    nx.draw_networkx_labels(G_block, pos_block, labels_block, font_size=7, font_family='sans-serif', font_weight='bold')

    plt.title("Transformer Block Deconstructed Into a Network of Pure Mathematical Primitives (H-DAG)", fontsize=14, fontweight='bold', pad=25)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("assets/nanogpt_atomic_block_graph.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved assets/nanogpt_atomic_block_graph.png")

if __name__ == "__main__":
    generate_visualizations()
