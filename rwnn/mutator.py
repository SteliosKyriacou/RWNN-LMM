import random
from collections import defaultdict, deque

def check_reachability(adj_out, src, dst):
    """Returns True if there is a path from src to dst."""
    visited = set()
    queue = deque([src])
    while queue:
        u = queue.popleft()
        if u == dst:
            return True
        for v in adj_out[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return False


def is_valid_dag(nodes_config, edges_config):
    """Verifies if the graph is a valid DAG (has no cycles and is fully connected)."""
    node_ids = {n['id'] for n in nodes_config}
    adj_out = defaultdict(list)
    adj_in = defaultdict(list)
    for u, v in edges_config:
        if u not in node_ids or v not in node_ids:
            return False
        adj_out[u].append(v)
        adj_in[v].append(u)

    # Kahn's algorithm cycle check
    in_degree = {n_id: 0 for n_id in node_ids}
    for u, v in edges_config:
        in_degree[v] += 1

    queue = deque([n_id for n_id, deg in in_degree.items() if deg == 0])
    visited_count = 0
    while queue:
        u = queue.popleft()
        visited_count += 1
        for v in adj_out[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return visited_count == len(node_ids)


def get_canonical_nano_gpt(vocab_size=1000, max_seq_len=256, d_model=128):
    """
    Constructs the canonical representation of a single-layer nanoGPT model as a DAG.
    """
    nodes = [
        {'id': 0, 'type': 'input', 'kwargs': {}},
        {'id': 1, 'type': 'token_embedding', 'kwargs': {'vocab_size': vocab_size, 'd_model': d_model}},
        {'id': 2, 'type': 'positional_embedding', 'kwargs': {'max_seq_len': max_seq_len, 'd_model': d_model}},
        {'id': 3, 'type': 'sum', 'kwargs': {}}, # Embeddings sum
        {'id': 4, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}},
        {'id': 5, 'type': 'causal_attention', 'kwargs': {'n_head': 4, 'd_model': d_model}}, # Q, K, V internally projected
        {'id': 6, 'type': 'sum', 'kwargs': {}}, # Attention residual addition
        {'id': 7, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}},
        {'id': 8, 'type': 'linear', 'kwargs': {'d_in': d_model, 'd_out': 4 * d_model}}, # MLP Expansion
        {'id': 9, 'type': 'activation', 'kwargs': {'act_type': 'gelu'}},
        {'id': 10, 'type': 'linear', 'kwargs': {'d_in': 4 * d_model, 'd_out': d_model}}, # MLP Contraction
        {'id': 11, 'type': 'sum', 'kwargs': {}}, # MLP residual addition
        {'id': 12, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}},
        {'id': 13, 'type': 'linear', 'kwargs': {'d_in': d_model, 'd_out': vocab_size}} # Output head
    ]

    edges = [
        (0, 1), # Input -> Token Embed
        (0, 2), # Input -> Position Embed
        (1, 3), # Token Embed -> Sum Embed
        (2, 3), # Position Embed -> Sum Embed
        
        (3, 4), # Embed Sum -> LN 1
        (4, 5), # LN 1 -> Attention
        
        (3, 6), # Residual: Embed Sum -> Sum Attn Resid
        (5, 6), # Attention -> Sum Attn Resid
        
        (6, 7), # Sum Attn Resid -> LN 2
        (7, 8), # LN 2 -> MLP Expansion
        (8, 9), # MLP Expansion -> Activation
        (9, 10), # Activation -> MLP Contraction
        
        (6, 11), # Residual: Sum Attn Resid -> Sum MLP Resid
        (10, 11), # MLP Contraction -> Sum MLP Resid
        
        (11, 12), # Sum MLP Resid -> LN Final
        (12, 13) # LN Final -> Output Head
    ]
    return nodes, edges


class GraphMutator:
    """Manages random mutations and crossovers of the RWNN H-DAG structures."""
    def __init__(self, vocab_size=1000, max_seq_len=256, d_model=128):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.node_types_pool = ['layer_norm', 'linear', 'causal_attention', 'activation', 'sum', 'concat', 'dropout']

    def generate_random_node(self, node_id):
        ntype = random.choice(self.node_types_pool)
        kwargs = {}
        if ntype == 'layer_norm':
            kwargs = {'d_model': self.d_model}
        elif ntype == 'linear':
            # Random expansion or projection to d_model
            d_out = random.choice([self.d_model, 2 * self.d_model, 4 * self.d_model])
            kwargs = {'d_in': self.d_model, 'd_out': d_out}
        elif ntype == 'causal_attention':
            # Number of heads must divide d_model
            possible_heads = [h for h in [2, 4, 8] if self.d_model % h == 0]
            n_head = random.choice(possible_heads) if possible_heads else 4
            kwargs = {'n_head': n_head, 'd_model': self.d_model}
        elif ntype == 'activation':
            kwargs = {'act_type': random.choice(['gelu', 'silu', 'relu'])}
        elif ntype == 'concat':
            kwargs = {'dim': -1}
        elif ntype == 'dropout':
            kwargs = {'dropout': 0.1}
        return {'id': node_id, 'type': ntype, 'kwargs': kwargs}

    def mutate_add_edge(self, nodes, edges):
        """Adds a random edge without introducing cycles."""
        node_ids = [n['id'] for n in nodes]
        if len(node_ids) < 2:
            return nodes, edges

        adj_out = defaultdict(list)
        for u, v in edges:
            adj_out[u].append(v)

        # Shuffle pairs to find a valid new edge
        pairs = [(u, v) for u in node_ids for v in node_ids if u != v and (u, v) not in edges]
        random.shuffle(pairs)

        for u, v in pairs:
            # Prevent invalid input/embedding connections
            if u == 0 and v not in {1, 2}:
                continue
            if v in {1, 2} and u != 0:
                continue
            # Prevent trivial cycles or connecting to input/embedding/outputs in invalid directions
            # Nodes 0, 1, 2 are inputs, node 13 (or terminal node) is output.
            # To be safe, verify that adding u -> v does NOT create reachability from v to u.
            if not check_reachability(adj_out, v, u):
                edges.append((u, v))
                break
        return nodes, edges

    def mutate_remove_edge(self, nodes, edges):
        """Removes a random edge, ensuring connectivity is preserved and no isolated nodes are created."""
        if len(edges) <= 2:
            return nodes, edges

        # Essential core path should not be severed completely
        random_edges = list(edges)
        random.shuffle(random_edges)

        adj_out = defaultdict(list)
        adj_in = defaultdict(list)
        for u, v in edges:
            adj_out[u].append(v)
            adj_in[v].append(u)

        for u, v in random_edges:
            # Check if deleting u -> v leaves either u or v completely stranded
            # (unless they are inputs/outputs)
            # Input nodes (0) have no in-edges. Output nodes have no out-edges.
            # Other nodes MUST have at least 1 in-edge and 1 out-edge.
            if len(adj_out[u]) > 1 and len(adj_in[v]) > 1:
                edges.remove((u, v))
                break
        return nodes, edges

    def mutate_add_node(self, nodes, edges):
        """Splices a new node onto an existing edge."""
        if not edges:
            return nodes, edges

        # Pick a random edge to split, excluding essential input/embedding boundaries
        splittable_edges = [e for e in edges if not (e[0] == 0 and e[1] in {1, 2})]
        if not splittable_edges:
            return nodes, edges

        edge_to_split = random.choice(splittable_edges)
        u, v = edge_to_split

        # Generate a unique node ID
        existing_ids = {n['id'] for n in nodes}
        new_id = max(existing_ids) + 1 if existing_ids else 0

        # Create random node
        new_node = self.generate_random_node(new_id)
        nodes.append(new_node)

        # Replace edge (u -> v) with (u -> w) and (w -> v)
        edges.remove((u, v))
        edges.append((u, new_id))
        edges.append((new_id, v))

        return nodes, edges

    def mutate_remove_node(self, nodes, edges):
        """Bypasses and removes a non-essential node."""
        essential_ids = {0, 1, 2, 3, 13} # Essential input/embedding/output head nodes from nanoGPT
        removable_nodes = [n for n in nodes if n['id'] not in essential_ids]
        if not removable_nodes:
            return nodes, edges

        target_node = random.choice(removable_nodes)
        w = target_node['id']

        adj_in = [u for u, v in edges if v == w]
        adj_out = [v for u, v in edges if u == w]

        # Bypass node w by connecting all its predecessors to all its successors
        new_edges = [(u, v) for u, v in edges if u != w and v != w]
        for u in adj_in:
            for v in adj_out:
                if (u, v) not in new_edges and u != v:
                    new_edges.append((u, v))

        # Check if the resulting graph is still a valid DAG
        test_nodes = [n for n in nodes if n['id'] != w]
        if is_valid_dag(test_nodes, new_edges):
            return test_nodes, new_edges

        # Fallback: if bypass fails or makes it invalid, return unchanged
        return nodes, edges

    def mutate(self, nodes, edges):
        """Applies a random structural mutation."""
        nodes_copy = [dict(n) for n in nodes]
        edges_copy = list(edges)

        mutation_type = random.choice(['add_node', 'add_edge', 'remove_edge', 'remove_node'])
        try:
            if mutation_type == 'add_node':
                nodes_copy, edges_copy = self.mutate_add_node(nodes_copy, edges_copy)
            elif mutation_type == 'add_edge':
                nodes_copy, edges_copy = self.mutate_add_edge(nodes_copy, edges_copy)
            elif mutation_type == 'remove_edge':
                nodes_copy, edges_copy = self.mutate_remove_edge(nodes_copy, edges_copy)
            elif mutation_type == 'remove_node':
                nodes_copy, edges_copy = self.mutate_remove_node(nodes_copy, edges_copy)
        except Exception:
            pass # Keep original on any unexpected error

        # Ensure we always return a valid, executing DAG
        if is_valid_dag(nodes_copy, edges_copy):
            return nodes_copy, edges_copy
        return nodes, edges

    def crossover(self, parent_a, parent_b):
        """
        Splat-join crossover of two H-DAG structures.
        A subset of nodes from both parents is merged, and edges are repaired.
        """
        nodes_a, edges_a = parent_a
        nodes_b, edges_b = parent_b

        # Essential nodes must be inherited from Parent A
        essential_ids = {0, 1, 2, 3, 13}
        essential_nodes = [n for n in nodes_a if n['id'] in essential_ids]

        # Pick random non-essential nodes from both parents
        non_ess_a = [n for n in nodes_a if n['id'] not in essential_ids]
        non_ess_b = [n for n in nodes_b if n['id'] not in essential_ids]

        # Shuffle and select a mix of nodes
        random.shuffle(non_ess_a)
        random.shuffle(non_ess_b)
        selected_nodes = non_ess_a[:len(non_ess_a)//2] + non_ess_b[:len(non_ess_b)//2]

        # Build clean merged node dictionary
        merged_nodes_dict = {n['id']: n for n in (essential_nodes + selected_nodes)}
        
        # Ensure unique keys and IDs
        # Re-index to avoid conflicts
        merged_nodes = list(merged_nodes_dict.values())
        merged_node_ids = {n['id'] for n in merged_nodes}

        # Keep parent edges that connect the surviving nodes
        all_edges = set(edges_a + edges_b)
        merged_edges = [(u, v) for u, v in all_edges if u in merged_node_ids and v in merged_node_ids]

        # Ensure it is a valid DAG; if not, fallback to Parent A
        if is_valid_dag(merged_nodes, merged_edges):
            return merged_nodes, merged_edges
        return nodes_a, edges_a


def get_gpt2_124m_dag(vocab_size=50257, max_seq_len=1024, d_model=768, n_layer=12, dropout=0.0):
    """
    Constructs the exact DAG configuration for a GPT-2 (124M) equivalent model.
    """
    nodes = [
        {'id': 0, 'type': 'input', 'kwargs': {}},
        {'id': 1, 'type': 'token_embedding', 'kwargs': {'vocab_size': vocab_size, 'd_model': d_model}},
        {'id': 2, 'type': 'positional_embedding', 'kwargs': {'max_seq_len': max_seq_len, 'd_model': d_model}},
        {'id': 3, 'type': 'sum', 'kwargs': {}} # Embeddings sum
    ]
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3)
    ]
    
    current_x = 3
    for l in range(n_layer):
        # Unique node IDs for Layer l
        ln1_id = 4 + l * 10
        attn_id = 5 + l * 10
        sum_attn_id = 6 + l * 10
        ln2_id = 7 + l * 10
        mlp_up_id = 8 + l * 10
        act_id = 9 + l * 10
        mlp_down_id = 10 + l * 10
        sum_mlp_id = 11 + l * 10
        
        # Add nodes
        nodes.append({'id': ln1_id, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}})
        nodes.append({'id': attn_id, 'type': 'causal_attention', 'kwargs': {'n_head': 12, 'd_model': d_model, 'dropout': dropout}})
        nodes.append({'id': sum_attn_id, 'type': 'sum', 'kwargs': {}})
        nodes.append({'id': ln2_id, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}})
        nodes.append({'id': mlp_up_id, 'type': 'linear', 'kwargs': {'d_in': d_model, 'd_out': 4 * d_model}})
        nodes.append({'id': act_id, 'type': 'activation', 'kwargs': {'act_type': 'gelu'}})
        nodes.append({'id': mlp_down_id, 'type': 'linear', 'kwargs': {'d_in': 4 * d_model, 'd_out': d_model}})
        nodes.append({'id': sum_mlp_id, 'type': 'sum', 'kwargs': {}})
        
        # Add edges
        # LN 1 & Attention Block
        edges.append((current_x, ln1_id))
        edges.append((ln1_id, attn_id))
        edges.append((current_x, sum_attn_id)) # Residual attention connection
        edges.append((attn_id, sum_attn_id))
        
        # LN 2 & MLP Block
        edges.append((sum_attn_id, ln2_id))
        edges.append((ln2_id, mlp_up_id))
        edges.append((mlp_up_id, act_id))
        edges.append((act_id, mlp_down_id))
        edges.append((sum_attn_id, sum_mlp_id)) # Residual MLP connection
        edges.append((mlp_down_id, sum_mlp_id))
        
        current_x = sum_mlp_id

    # Output head blocks
    ln_f_id = 4 + n_layer * 10
    head_id = 13 # Output head is always 13 for consistent return head
    
    nodes.append({'id': ln_f_id, 'type': 'layer_norm', 'kwargs': {'d_model': d_model}})
    nodes.append({'id': head_id, 'type': 'linear', 'kwargs': {'d_in': d_model, 'd_out': vocab_size}})
    
    edges.append((current_x, ln_f_id))
    edges.append((ln_f_id, head_id))
    
    return nodes, edges
