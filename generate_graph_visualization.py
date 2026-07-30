import os
import networkx as nx
import matplotlib.pyplot as plt
from rwnn.mutator import get_canonical_nano_gpt, get_gpt2_dag

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
        'activation': '#e6ccff',  # Soft Purple
        'matmul': '#ff9999',      # Soft Red
        'add_bias': '#ffcc99',    # Light Orange
        'transpose': '#b3f0ff',   # Cyan
        'reshape': '#ffd1b3',     # Light Orange
        'causal_batch_matmul': '#ffb3d1' # Light Pink
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

    node_colors_atomic = [color_map.get(G_atomic.nodes[n]['type'], '#cccccc') for n in G_atomic.nodes]
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
    block_nodes = [
        {'id': 0, 'type': 'input'},
        {'id': 1, 'type': 'token_embedding'},
        {'id': 2, 'type': 'positional_embedding'},
        {'id': 3, 'type': 'sum'},
        
        # LayerNorm 1
        {'id': 4, 'type': 'mean_reduce'},
        {'id': 5, 'type': 'subtract'},
        {'id': 6, 'type': 'square'},
        {'id': 7, 'type': 'mean_reduce'},
        {'id': 8, 'type': 'sqrt'},
        {'id': 9, 'type': 'divide'},
        {'id': 10, 'type': 'scale_shift'},
        
        # Q, K, V
        {'id': 11, 'type': 'matmul'},
        {'id': 12, 'type': 'add_bias'},
        {'id': 13, 'type': 'matmul'},
        {'id': 14, 'type': 'add_bias'},
        {'id': 15, 'type': 'matmul'},
        {'id': 16, 'type': 'add_bias'},
        
        # Reshape/Transpose
        {'id': 17, 'type': 'reshape'},
        {'id': 18, 'type': 'transpose'},
        {'id': 19, 'type': 'reshape'},
        {'id': 20, 'type': 'transpose'},
        {'id': 21, 'type': 'reshape'},
        {'id': 22, 'type': 'transpose'},
        
        # Attention
        {'id': 23, 'type': 'causal_batch_matmul'},
        {'id': 24, 'type': 'activation'},
        {'id': 25, 'type': 'causal_batch_matmul'},
        
        # Output
        {'id': 26, 'type': 'transpose'},
        {'id': 27, 'type': 'reshape'},
        {'id': 28, 'type': 'matmul'},
        {'id': 29, 'type': 'add_bias'},
        
        # Residual Sum
        {'id': 30, 'type': 'sum'},
        
        # MLP Block
        {'id': 31, 'type': 'matmul'},
        {'id': 32, 'type': 'add_bias'},
        {'id': 33, 'type': 'activation'},
        {'id': 34, 'type': 'matmul'},
        {'id': 35, 'type': 'add_bias'},
        
        # Output layers
        {'id': 36, 'type': 'sum'},
        {'id': 37, 'type': 'layer_norm'},
        {'id': 38, 'type': 'linear'}
    ]

    block_edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (3, 4), (3, 5), (4, 5), (5, 6), (6, 7), (7, 8), (5, 9), (8, 9), (9, 10),
        (10, 11), (11, 12), (10, 13), (13, 14), (10, 15), (15, 16),
        (12, 17), (17, 18), (14, 19), (19, 20), (16, 21), (21, 22),
        (18, 23), (20, 23), (23, 24), (24, 25), (22, 25),
        (25, 26), (26, 27), (27, 28), (28, 29),
        (3, 30), (29, 30),
        (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
        (30, 36), (35, 36), (36, 37), (37, 38)
    ]

    G_block = nx.DiGraph()
    for n in block_nodes:
        G_block.add_node(n['id'], label=f"{n['id']}: {n['type']}", type=n['type'])
    G_block.add_edges_from(block_edges)

    pos_block = {
        0: (0, 15), 1: (-3, 13.5), 2: (3, 13.5), 3: (0, 12),
        4: (-2.5, 10.5), 5: (0, 9.5), 6: (2.5, 8.5), 7: (2.5, 7.2), 8: (2.5, 6.0), 9: (0, 5.0), 10: (0, 3.8),
        11: (-5, 2), 12: (-5, 0.8), 13: (0, 2), 14: (0, 0.8), 15: (5, 2), 16: (5, 0.8),
        17: (-5, -0.4), 18: (-5, -1.6), 19: (0, -0.4), 20: (0, -1.6), 21: (5, -0.4), 22: (5, -1.6),
        23: (-2.5, -2.8), 24: (-2.5, -4.0), 25: (2.5, -4.0),
        26: (2.5, -5.2), 27: (2.5, -6.4), 28: (0, -7.5), 29: (0, -8.7),
        30: (0, -10.0),
        31: (-4, -11.5), 32: (-4, -12.5), 33: (-4, -13.5), 34: (-4, -14.5), 35: (-4, -15.5),
        36: (0, -17.0), 37: (0, -18.2), 38: (0, -19.5)
    }

    node_colors_block = [color_map.get(G_block.nodes[n]['type'], '#cccccc') for n in G_block.nodes]
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


    # ==================== PLOTS 4-7: All GPT-2 Model Level Layouts ====================
    # Generalised Multi-layer Layout Plotter
    def plot_gpt2_level_graph(model_type, filename, title, figsize):
        print(f"Generating full graph layout for {model_type}...")
        nodes, edges = get_gpt2_dag(model_type)
        
        G = nx.DiGraph()
        for n in nodes:
            # Short clean labels
            label = f"{n['id']}:{n['type'].replace('token_embedding', 'WTE').replace('positional_embedding', 'WPE').replace('causal_attention', 'SelfAttn').replace('layer_norm', 'LN').replace('linear', 'MLP').replace('activation', 'GELU')}"
            G.add_node(n['id'], label=label, type=n['type'])
        G.add_edges_from(edges)
        
        # Position Coordinates
        pos = {
            0: (0, 0),       # Input Token Nodes
            1: (-2, -1.2),   # WTE
            2: (2, -1.2),    # WPE
            3: (0, -2.4)     # Embed Sum
        }
        
        # Find number of layers
        n_layer = (len(nodes) - 6) // 8
        
        # Layer Blocks
        for l in range(n_layer):
            ln1 = 4 + l * 10
            attn = 5 + l * 10
            sum_attn = 6 + l * 10
            ln2 = 7 + l * 10
            mlp_up = 8 + l * 10
            act = 9 + l * 10
            mlp_down = 10 + l * 10
            sum_mlp = 11 + l * 10
            
            y_base = -3.5 - l * 5.0
            pos[ln1] = (-2, y_base)
            pos[attn] = (-2, y_base - 1.2)
            pos[sum_attn] = (-2, y_base - 2.4)
            
            pos[ln2] = (2, y_base)
            pos[mlp_up] = (2, y_base - 1.0)
            pos[act] = (2, y_base - 2.0)
            pos[mlp_down] = (2, y_base - 3.0)
            
            pos[sum_mlp] = (0, y_base - 4.2)
            
        # Final head
        ln_f = 4 + n_layer * 10
        head = 13
        
        pos[ln_f] = (0, -3.5 - n_layer * 5.0)
        pos[head] = (0, -3.5 - n_layer * 5.0 - 1.5)
        
        # Render
        plt.figure(figsize=figsize)
        node_colors = [color_map.get(G.nodes[n]['type'], '#cccccc') for n in G.nodes]
        labels = nx.get_node_attributes(G, 'label')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1100, edgecolors='#333333', linewidths=0.8)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='#888888', width=0.8)
        nx.draw_networkx_labels(G, pos, labels, font_size=5, font_family='sans-serif', font_weight='bold')
        
        plt.title(title, fontsize=13, fontweight='bold', pad=15)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {filename}")

    # Generate all four model layouts dynamically!
    plot_gpt2_level_graph('toy', 'assets/gpt2_toy_6l_layout.png', "Toy GPT-2 Configuration (6 Layers, 10.8M parameters)", (8, 18))
    plot_gpt2_level_graph('gpt2', 'assets/gpt2_124m_12l_layout.png', "Standard GPT-2 Configuration (12 Layers, 124M parameters)", (8, 30))
    plot_gpt2_level_graph('gpt2-medium', 'assets/gpt2_medium_24l_layout.png', "GPT-2 Medium Configuration (24 Layers, 350M parameters)", (10, 50))
    
    # We omit larger ones or render with extra size to avoid memory limit issues
    plot_gpt2_level_graph('gpt2-large', 'assets/gpt2_large_36l_layout.png', "GPT-2 Large Configuration (36 Layers, 774M parameters)", (12, 70))
    # plot_gpt2_level_graph('gpt2-xl', 'assets/gpt2_xl_48l_layout.png', "GPT-2 XL Configuration (48 Layers, 1.5B parameters)", (14, 90))

if __name__ == "__main__":
    generate_visualizations()
