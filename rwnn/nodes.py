import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RWNNNode(nn.Module):
    """Base class for all heterogeneous nodes in the RWNN Graph."""
    def __init__(self, node_id, node_type):
        super().__init__()
        self.node_id = node_id
        self.node_type = node_type

    def expected_input_dim(self, input_index=0):
        """
        Returns the expected dimension (last axis) for the input at input_index.
        Returns None if any dimension is accepted or handled dynamically.
        """
        return None

    def forward(self, inputs):
        """
        inputs: list of torch.Tensor
        Returns: torch.Tensor
        """
        raise NotImplementedError


class TokenEmbeddingNode(RWNNNode):
    def __init__(self, node_id, vocab_size, d_model):
        super().__init__(node_id, "token_embedding")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.wte = nn.Embedding(vocab_size, d_model)

    def expected_input_dim(self, input_index=0):
        return None # Expects integer token indices, not embedding vectors

    def forward(self, inputs):
        # Expects inputs[0] to be integer tensor of shape [B, T]
        x = inputs[0]
        return self.wte(x)


class PositionalEmbeddingNode(RWNNNode):
    def __init__(self, node_id, max_seq_len, d_model):
        super().__init__(node_id, "positional_embedding")
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.wpe = nn.Embedding(max_seq_len, d_model)

    def expected_input_dim(self, input_index=0):
        return None # Can be generated from sequence length of any input

    def forward(self, inputs):
        # inputs[0] can be token indices [B, T] or hidden state [B, T, D]
        # We just need its sequence length T to generate positions
        x = inputs[0]
        t = x.size(1)
        device = x.device
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # [1, T]
        return self.wpe(pos)


class LayerNormNode(RWNNNode):
    def __init__(self, node_id, d_model, eps=1e-5):
        super().__init__(node_id, "layer_norm")
        self.d_model = d_model
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def expected_input_dim(self, input_index=0):
        return self.d_model

    def forward(self, inputs):
        return self.ln(inputs[0])


class LinearNode(RWNNNode):
    def __init__(self, node_id, d_in, d_out, bias=True):
        super().__init__(node_id, "linear")
        self.d_in = d_in
        self.d_out = d_out
        self.linear = nn.Linear(d_in, d_out, bias=bias)

    def expected_input_dim(self, input_index=0):
        return self.d_in

    def forward(self, inputs):
        return self.linear(inputs[0])


class CausalAttentionNode(RWNNNode):
    def __init__(self, node_id, n_head, d_model, dropout=0.0):
        super().__init__(node_id, "causal_attention")
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.head_dim = d_model // n_head

        # Internal projections for standard self-contained mode (if 1 input is provided)
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=True)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def expected_input_dim(self, input_index=0):
        return self.d_model

    def forward(self, inputs):
        # Supports:
        # 1. Self-contained: inputs[0] is the hidden state. Q, K, V computed internally.
        # 2. Split inputs: inputs[0]=Q, inputs[1]=K, inputs[2]=V.
        if len(inputs) == 1:
            x = inputs[0]
            B, T, C = x.size()
            # Calculate query, key, values
            q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        elif len(inputs) >= 3:
            q, k, v = inputs[0], inputs[1], inputs[2]
            B, T, C = q.size()
        else:
            # Fallback if 2 inputs: reuse first for Q, second for K and V
            q = inputs[0]
            k = v = inputs[1]
            B, T, C = q.size()

        # Reshape to [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # [B, n_head, T, head_dim]
        
        # Re-assemble head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class ActivationNode(RWNNNode):
    def __init__(self, node_id, act_type="gelu"):
        super().__init__(node_id, f"activation_{act_type}")
        self.act_type = act_type

    def forward(self, inputs):
        x = inputs[0]
        if self.act_type == "gelu":
            return F.gelu(x)
        elif self.act_type == "silu":
            return F.silu(x)
        elif self.act_type == "relu":
            return F.relu(x)
        else:
            return x


class SumNode(RWNNNode):
    def __init__(self, node_id):
        super().__init__(node_id, "sum")

    def forward(self, inputs):
        # Element-wise addition of all inputs
        if len(inputs) == 0:
            raise ValueError("SumNode requires at least one input.")
        out = inputs[0]
        for i in range(1, len(inputs)):
            out = out + inputs[i]
        return out


class ConcatNode(RWNNNode):
    def __init__(self, node_id, dim=-1):
        super().__init__(node_id, "concat")
        self.dim = dim

    def forward(self, inputs):
        if len(inputs) == 0:
            raise ValueError("ConcatNode requires at least one input.")
        return torch.cat(inputs, dim=self.dim)


class ElementMulNode(RWNNNode):
    def __init__(self, node_id):
        super().__init__(node_id, "element_mul")

    def forward(self, inputs):
        if len(inputs) == 0:
            raise ValueError("ElementMulNode requires at least one input.")
        out = inputs[0]
        for i in range(1, len(inputs)):
            out = out * inputs[i]
        return out


class DropoutNode(RWNNNode):
    def __init__(self, node_id, dropout=0.0):
        super().__init__(node_id, "dropout")
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return self.dropout(inputs[0])


class MeanReduceNode(RWNNNode):
    def __init__(self, node_id, dim=-1):
        super().__init__(node_id, "mean_reduce")
        self.dim = dim

    def forward(self, inputs):
        return inputs[0].mean(dim=self.dim, keepdim=True)


class SquareNode(RWNNNode):
    def __init__(self, node_id):
        super().__init__(node_id, "square")

    def forward(self, inputs):
        return torch.square(inputs[0])


class SubtractNode(RWNNNode):
    def __init__(self, node_id):
        super().__init__(node_id, "subtract")

    def forward(self, inputs):
        return inputs[0] - inputs[1]


class DivideNode(RWNNNode):
    def __init__(self, node_id):
        super().__init__(node_id, "divide")

    def forward(self, inputs):
        return inputs[0] / inputs[1]


class SqrtNode(RWNNNode):
    def __init__(self, node_id, eps=1e-5):
        super().__init__(node_id, "sqrt")
        self.eps = eps

    def forward(self, inputs):
        return torch.sqrt(inputs[0] + self.eps)


class ScaleShiftNode(RWNNNode):
    def __init__(self, node_id, d_model):
        super().__init__(node_id, "scale_shift")
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def expected_input_dim(self, input_index=0):
        return self.d_model

    def forward(self, inputs):
        return inputs[0] * self.gamma + self.beta


class MatMulNode(RWNNNode):
    def __init__(self, node_id, d_in, d_out):
        super().__init__(node_id, "matmul")
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * (1.0 / (d_in ** 0.5)))

    def expected_input_dim(self, input_index=0):
        return self.d_in

    def forward(self, inputs):
        return F.linear(inputs[0], self.weight)


class AddBiasNode(RWNNNode):
    def __init__(self, node_id, d_model):
        super().__init__(node_id, "add_bias")
        self.d_model = d_model
        self.bias = nn.Parameter(torch.zeros(d_model))

    def expected_input_dim(self, input_index=0):
        return self.d_model

    def forward(self, inputs):
        return inputs[0] + self.bias


class TransposeNode(RWNNNode):
    def __init__(self, node_id, dim1, dim2):
        super().__init__(node_id, "transpose")
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, inputs):
        return inputs[0].transpose(self.dim1, self.dim2)


class ReshapeNode(RWNNNode):
    def __init__(self, node_id, shape):
        super().__init__(node_id, "reshape")
        self.shape = shape

    def forward(self, inputs):
        return inputs[0].reshape(*self.shape)


class CausalBatchMatMulNode(RWNNNode):
    def __init__(self, node_id, scale=1.0):
        super().__init__(node_id, "causal_batch_matmul")
        self.scale = scale

    def forward(self, inputs):
        q = inputs[0]
        k = inputs[1]
        att = (q @ k.transpose(-2, -1)) * self.scale
        T = q.size(-2)
        mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        return att


class MoEFFNNode(RWNNNode):
    """Fused Mixture-of-Experts feed-forward block.

    router -> top-k experts -> gate-weighted combine. Each expert is a standard
    2-layer FFN (Linear -> activation -> Linear). Real sparse dispatch: every token
    is only run through its top_k experts, so wall-clock and active-FLOPs scale with
    top_k, while capacity (and memory) scale with n_experts. n_experts == 1 collapses
    to a plain dense FFN. Exposes `.aux_loss` (Switch-style load balance) after each
    forward; RWNNGraph.moe_aux_loss() collects it for the training objective.
    """
    def __init__(self, node_id, d_model, n_experts=4, top_k=2, d_hidden=None,
                 act_type="gelu", dropout=0.0):
        super().__init__(node_id, "moe_ffn")
        self.d_model = d_model
        self.n_experts = int(n_experts)
        self.top_k = min(int(top_k), self.n_experts)
        self.d_hidden = int(d_hidden) if d_hidden else 4 * d_model
        self.act_type = act_type
        self.router = nn.Linear(d_model, self.n_experts, bias=False)
        # Stacked expert weights: [E, d_in, d_out]
        self.w_in = nn.Parameter(torch.empty(self.n_experts, d_model, self.d_hidden))
        self.w_out = nn.Parameter(torch.empty(self.n_experts, self.d_hidden, d_model))
        self.b_in = nn.Parameter(torch.zeros(self.n_experts, self.d_hidden))
        self.b_out = nn.Parameter(torch.zeros(self.n_experts, d_model))
        nn.init.normal_(self.w_in, std=0.02)
        nn.init.normal_(self.w_out, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        self.aux_loss = self._zero

    def expected_input_dim(self, input_index=0):
        return self.d_model

    def _act(self, x):
        if self.act_type == "silu":
            return F.silu(x)
        if self.act_type == "relu":
            return F.relu(x)
        return F.gelu(x)

    def forward(self, inputs):
        x = inputs[0]                                   # [B, T, d]
        B, T, d = x.shape
        xf = x.reshape(-1, d)                            # [N, d]
        N = xf.shape[0]
        probs = F.softmax(self.router(xf), dim=-1)       # [N, E]
        topv, topi = probs.topk(self.top_k, dim=-1)      # [N, k]
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)   # renormalise gates

        out = torch.zeros_like(xf)
        for e in range(self.n_experts):
            sel = (topi == e)                            # [N, k] this expert selected in some slot
            tok = sel.any(dim=-1)                         # [N] tokens using expert e
            if not bool(tok.any()):
                continue
            gate = (topv * sel).sum(dim=-1)[tok]          # [n_e] combined gate for expert e
            xe = xf[tok]                                  # [n_e, d]  (only routed tokens -> sparse compute)
            he = self._act(xe @ self.w_in[e] + self.b_in[e])
            ye = he @ self.w_out[e] + self.b_out[e]
            out[tok] = out[tok] + gate[:, None] * ye
        out = self.drop(out).reshape(B, T, d)

        # Switch-style load-balance aux loss: E * sum_e (f_e * P_e)
        importance = probs.mean(dim=0)                   # [E] mean router prob
        top1 = topi[:, 0]
        frac = torch.bincount(top1, minlength=self.n_experts).float() / max(N, 1)  # [E]
        self.aux_loss = self.n_experts * (frac * importance).sum()
        return out
