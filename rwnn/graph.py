import torch
import torch.nn as nn
from collections import defaultdict, deque
from rwnn.nodes import (
    TokenEmbeddingNode, PositionalEmbeddingNode, LayerNormNode,
    LinearNode, CausalAttentionNode, ActivationNode, SumNode,
    ConcatNode, ElementMulNode, DropoutNode, RWNNNode,
    MeanReduceNode, SquareNode, SubtractNode, DivideNode,
    SqrtNode, ScaleShiftNode, MatMulNode, AddBiasNode,
    TransposeNode, ReshapeNode, CausalBatchMatMulNode, MoEFFNNode, SoftmaxNode, SliceNode
)

class EdgeConnection(nn.Module):
    """Dynamic Edge Connection that automatically projects feature dimensions if they mismatch."""
    def __init__(self, d_src, d_tgt):
        super().__init__()
        self.d_src = d_src
        self.d_tgt = d_tgt
        if d_src is not None and d_tgt is not None and d_src != d_tgt:
            self.proj = nn.Linear(d_src, d_tgt, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        if not torch.is_floating_point(x):
            return x # Pass through token indices/ints
        return self.proj(x)


class RWNNGraph(nn.Module):
    """
    Topologically compiled RWNN Graph executing heterogeneous tensor operations.
    Fully vectorizable and compatible with PyTorch autograd.
    """
    def __init__(self, nodes_config, edges_config, global_d_model=128):
        """
        nodes_config: list of dicts:
            e.g. [{'id': 0, 'type': 'input', 'kwargs': {}},
                  {'id': 1, 'type': 'token_embedding', 'kwargs': {'vocab_size': 1000, 'd_model': 128}}]
        edges_config: list of tuples:
            e.g. [(0, 1), (1, 2)]
        global_d_model: default fallback dimension for alignment.
        """
        super().__init__()
        self.nodes_config = {n['id']: n for n in nodes_config}
        self.edges_config = edges_config
        self.global_d_model = global_d_model

        # 1. Instantiate the individual node modules
        self.nodes = nn.ModuleDict()
        for n in nodes_config:
            self.nodes[str(n['id'])] = self._create_node(n)

        # 2. Build adjacency mappings
        self.adj_in = defaultdict(list)
        self.adj_out = defaultdict(list)
        for u, v in edges_config:
            self.adj_in[v].append(u)
            self.adj_out[u].append(v)

        # 3. Find topological execution order (Kahn's algorithm)
        self.execution_order = self._topological_sort()

        # 4. Trace feature dimensions and compile Edge Connections
        self.edges = nn.ModuleDict()
        self.node_out_dims = {} # Trace node ID -> output feature dimension
        self._trace_dimensions_and_compile_edges()

    def _create_node(self, n):
        n_id = n['id']
        n_type = n['type']
        kwargs = n.get('kwargs', {})

        if n_type == 'input':
            # Dummy node to hold raw input
            class InputPlaceholderNode(RWNNNode):
                def __init__(self, nid):
                    super().__init__(nid, "input")
                def forward(self, inputs):
                    return inputs[0]
            return InputPlaceholderNode(n_id)
        elif n_type == 'token_embedding':
            return TokenEmbeddingNode(n_id, **kwargs)
        elif n_type == 'positional_embedding':
            return PositionalEmbeddingNode(n_id, **kwargs)
        elif n_type == 'layer_norm':
            return LayerNormNode(n_id, **kwargs)
        elif n_type == 'linear':
            return LinearNode(n_id, **kwargs)
        elif n_type == 'causal_attention':
            return CausalAttentionNode(n_id, **kwargs)
        elif n_type == 'activation':
            return ActivationNode(n_id, **kwargs)
        elif n_type == 'sum':
            return SumNode(n_id)
        elif n_type == 'concat':
            return ConcatNode(n_id, **kwargs)
        elif n_type == 'element_mul':
            return ElementMulNode(n_id)
        elif n_type == 'dropout':
            return DropoutNode(n_id, **kwargs)
        elif n_type == 'mean_reduce':
            return MeanReduceNode(n_id, **kwargs)
        elif n_type == 'square':
            return SquareNode(n_id)
        elif n_type == 'subtract':
            return SubtractNode(n_id)
        elif n_type == 'divide':
            return DivideNode(n_id)
        elif n_type == 'sqrt':
            return SqrtNode(n_id, **kwargs)
        elif n_type == 'scale_shift':
            return ScaleShiftNode(n_id, **kwargs)
        elif n_type == 'matmul':
            return MatMulNode(n_id, **kwargs)
        elif n_type == 'add_bias':
            return AddBiasNode(n_id, **kwargs)
        elif n_type == 'transpose':
            return TransposeNode(n_id, **kwargs)
        elif n_type == 'reshape':
            return ReshapeNode(n_id, **kwargs)
        elif n_type == 'causal_batch_matmul':
            return CausalBatchMatMulNode(n_id, **kwargs)
        elif n_type == 'moe_ffn':
            return MoEFFNNode(n_id, **kwargs)
        elif n_type == 'softmax':
            return SoftmaxNode(n_id, **kwargs)
        elif n_type == 'slice':
            return SliceNode(n_id, **kwargs)
        else:
            raise ValueError(f"Unknown node type: {n_type}")

    def _topological_sort(self):
        in_degree = {n_id: 0 for n_id in self.nodes_config.keys()}
        for u, v in self.edges_config:
            in_degree[v] += 1

        queue = deque([n_id for n_id, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.adj_out[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(order) < len(self.nodes_config):
            raise ValueError("Graph contains cycles or disconnected components!")
        return order

    def _trace_dimensions_and_compile_edges(self):
        """Topologically traces output dimensions of each node to compile EdgeConnections."""
        for u in self.execution_order:
            node = self.nodes[str(u)]
            predecessors = self.adj_in[u]

            # Determine expected target dimension for node u
            if not predecessors:
                if u == 0:
                    self.node_out_dims[u] = None # None means raw token indices/placeholder
                    continue
                else:
                    d_tgt = self.global_d_model
            else:
                d_tgt = node.expected_input_dim(0)
                if d_tgt is None:
                    # Fallback to output dimension of first predecessor, or global_d_model
                    first_pred_dim = self.node_out_dims[predecessors[0]]
                    d_tgt = first_pred_dim if first_pred_dim is not None else self.global_d_model

                # Setup edge projections for all incoming connections
                for i, p in enumerate(predecessors):
                    d_src = self.node_out_dims[p]
                    
                    # If predecessors have multiple different expected target dimensions (like ConcatNode which gathers all)
                    if isinstance(node, ConcatNode):
                        d_tgt_i = d_src if d_src is not None else self.global_d_model
                    elif hasattr(node, 'expected_input_dim'):
                        d_tgt_i = node.expected_input_dim(i)
                        if d_tgt_i is None:
                            d_tgt_i = d_tgt
                    else:
                        d_tgt_i = d_tgt

                    # Global atomic broadcasting bypass:
                    # If the source node outputs 1, and the node does not explicitly require a fixed dimension != 1,
                    # we let it broadcast by setting d_tgt_i = 1.
                    if d_src == 1:
                        explicit_dim = node.expected_input_dim(i) if hasattr(node, 'expected_input_dim') else None
                        if explicit_dim is None:
                            d_tgt_i = 1

                    edge_conn = EdgeConnection(d_src, d_tgt_i)
                    self.edges[f"{p}_{u}"] = edge_conn

            # Determine output dimension of node u
            if isinstance(node, TokenEmbeddingNode):
                self.node_out_dims[u] = node.d_model
            elif isinstance(node, PositionalEmbeddingNode):
                self.node_out_dims[u] = node.d_model
            elif isinstance(node, LayerNormNode):
                self.node_out_dims[u] = node.d_model
            elif isinstance(node, LinearNode):
                self.node_out_dims[u] = node.d_out
            elif isinstance(node, CausalAttentionNode):
                self.node_out_dims[u] = node.d_model
            elif isinstance(node, MoEFFNNode):
                self.node_out_dims[u] = node.d_model
            elif isinstance(node, ActivationNode):
                pred_dim = self.node_out_dims[predecessors[0]] if predecessors else self.global_d_model
                self.node_out_dims[u] = pred_dim if pred_dim is not None else self.global_d_model
            elif isinstance(node, SumNode):
                self.node_out_dims[u] = d_tgt
            elif isinstance(node, ConcatNode):
                total_dim = 0
                for p in predecessors:
                    p_dim = self.node_out_dims[p]
                    total_dim += p_dim if p_dim is not None else self.global_d_model
                self.node_out_dims[u] = total_dim if predecessors else self.global_d_model
            elif isinstance(node, ElementMulNode):
                self.node_out_dims[u] = d_tgt
            elif isinstance(node, DropoutNode):
                pred_dim = self.node_out_dims[predecessors[0]] if predecessors else self.global_d_model
                self.node_out_dims[u] = pred_dim if pred_dim is not None else self.global_d_model
            elif isinstance(node, SoftmaxNode):
                pred_dim = self.node_out_dims[predecessors[0]] if predecessors else self.global_d_model
                self.node_out_dims[u] = pred_dim if pred_dim is not None else self.global_d_model
            elif isinstance(node, SliceNode):
                self.node_out_dims[u] = node.end - node.start
            elif isinstance(node, MeanReduceNode):
                pred_dim = self.node_out_dims[predecessors[0]] if predecessors else self.global_d_model
                self.node_out_dims[u] = 1 if node.dim == -1 else pred_dim
            elif isinstance(node, (SquareNode, SubtractNode, DivideNode, SqrtNode)):
                pred_dim = self.node_out_dims[predecessors[0]] if predecessors else self.global_d_model
                self.node_out_dims[u] = pred_dim
            elif isinstance(node, ScaleShiftNode):
                self.node_out_dims[u] = node.d_model
            elif isinstance(node, MatMulNode):
                self.node_out_dims[u] = node.d_out
            elif isinstance(node, AddBiasNode):
                self.node_out_dims[u] = node.d_model
            elif isinstance(node, (TransposeNode, ReshapeNode, CausalBatchMatMulNode)):
                pred_dim = self.node_out_dims[predecessors[0]] if predecessors else self.global_d_model
                self.node_out_dims[u] = pred_dim
            else:
                self.node_out_dims[u] = d_tgt

    def forward(self, x):
        """
        x: input tensor of token indices [B, T] (or hidden states if first node is not embedding)
        Returns: logits or output tensor of final node
        """
        outputs = {}
        for u in self.execution_order:
            node = self.nodes[str(u)]
            predecessors = self.adj_in[u]

            if not predecessors:
                if u == 0:
                    # Raw input token indices node
                    outputs[u] = node([x])
                else:
                    # Disconnected non-input node: feed dummy float tensor of shape [B, T, global_d_model]
                    B, T = x.size()
                    dummy = torch.zeros(B, T, self.global_d_model, device=x.device, dtype=torch.float32)
                    outputs[u] = node([dummy])
            else:
                # Retrieve predecessor outputs and apply edge connections
                node_inputs = []
                for p in predecessors:
                    p_out = outputs[p]
                    edge_conn = self.edges[f"{p}_{u}"]
                    node_inputs.append(edge_conn(p_out))
                
                outputs[u] = node(node_inputs)

        # We want to return the output of the language model head (Node 13) if present,
        # otherwise fallback to the last node in topological order.
        if 13 in outputs:
            return outputs[13]
        terminal_nodes = [u for u in self.execution_order if not self.adj_out[u]]
        if not terminal_nodes:
            raise ValueError("Graph has no terminal/output node!")
        return outputs[terminal_nodes[-1]]

    def moe_aux_loss(self):
        """Sum the load-balance aux losses of all MoE nodes from the most recent forward.
        Returns a scalar tensor (0.0 if there are no MoE nodes)."""
        total = None
        for m in self.modules():
            if isinstance(m, MoEFFNNode):
                total = m.aux_loss if total is None else total + m.aux_loss
        if total is None:
            p = next(self.parameters(), None)
            return torch.zeros((), device=p.device if p is not None else 'cpu')
        return total
