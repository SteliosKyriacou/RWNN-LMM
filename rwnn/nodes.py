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
    """Mean over `dim`. Over the FEATURE axis it is a plain per-token mean. Over the TOKEN axis
    (dim==1) it is a CAUSAL prefix/running mean — position t averages only tokens <= t — so it can
    never pool information from the future into the present (no autoregressive leakage)."""
    def __init__(self, node_id, dim=-1):
        super().__init__(node_id, "mean_reduce")
        self.dim = dim

    def forward(self, inputs):
        x = inputs[0]
        d = self.dim if self.dim >= 0 else x.dim() + self.dim
        if d == 1 and x.dim() >= 2:                       # token axis -> causal prefix mean (no future leak)
            csum = x.cumsum(dim=1)
            cnt = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype)
            cnt = cnt.view([1, x.size(1)] + [1] * (x.dim() - 2))
            return csum / cnt
        return x.mean(dim=self.dim, keepdim=True)


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


class SoftmaxNode(RWNNNode):
    """Softmax over `dim`. Over the FEATURE axis it is an ordinary softmax. Over the TOKEN axis (dim==1)
    it is a CAUSAL PREFIX softmax: position t is normalized only over tokens <= t (the denominator is a
    running sum), so it can never depend on future tokens. Safe because it keeps token-index == time."""
    def __init__(self, node_id, dim=-1):
        super().__init__(node_id, "softmax")
        self.dim = dim

    def forward(self, inputs):
        x = inputs[0]
        d = self.dim if self.dim >= 0 else x.dim() + self.dim
        if d == 1 and x.dim() >= 2:                       # token axis -> causal prefix softmax (no future leak)
            e = torch.exp(x.clamp(-30.0, 30.0))
            return e / (e.cumsum(dim=1) + 1e-9)
        return F.softmax(x, dim=self.dim)


class SliceNode(RWNNNode):
    """Select channels [start:end] of the last dimension. Primitive atom: lets a multi-way gate be
    split into per-expert scalars (linear->softmax over E, then slice(i,i+1) -> element_mul expert i)."""
    def __init__(self, node_id, start=0, end=1):
        super().__init__(node_id, "slice")
        self.start = int(start)
        self.end = int(end)

    def forward(self, inputs):
        return inputs[0][..., self.start:self.end]


# ---- Data-movement / dispatch primitives (data-dependent indexing) --------------------------------
class TopKNode(RWNNNode):
    """Keep the top-k values along the last dim (others -> 0), renormalized so the kept weights sum to 1.
    The selection atom for sparse routing: e.g. linear(->E) -> softmax -> top_k gives a sparse gate."""
    def __init__(self, node_id, k=2):
        super().__init__(node_id, "top_k")
        self.k = int(k)

    def forward(self, inputs):
        x = inputs[0]
        k = min(self.k, x.size(-1))
        v, i = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x).scatter(-1, i, v)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-9)


class GatherNode(RWNNNode):
    """Data-dependent row gather: select positions of inputs[0] using integer indices inputs[1] along
    `dim` (default the token axis 1). Produces a (possibly smaller) tensor -> the basis of real sparse
    dispatch, because a gathered subset flowing into a `linear` makes that linear compute fewer rows."""
    def __init__(self, node_id, dim=1):
        super().__init__(node_id, "gather")
        self.dim = dim

    def forward(self, inputs):
        data, idx = inputs[0], inputs[1].reshape(-1).long()
        return torch.index_select(data, self.dim, idx)


class ScatterAddNode(RWNNNode):
    """Scatter-add: add src (inputs[2]) into a zeros-like(inputs[0]) target at integer indices
    inputs[1] along `dim`. Recombines routed/gathered expert outputs back to full sequence positions."""
    def __init__(self, node_id, dim=1):
        super().__init__(node_id, "scatter_add")
        self.dim = dim

    def forward(self, inputs):
        target, idx, src = inputs[0], inputs[1].reshape(-1).long(), inputs[2]
        out = torch.zeros_like(target)
        out.index_add_(self.dim, idx, src)
        return out


class ScanNode(RWNNNode):
    """Causal gated linear SCAN — the recurrence / STATE primitive (the SSM / linear-attention / RWKV /
    Mamba family). Sweeps left-to-right maintaining a running state:
        h_t = f_t * h_{t-1} + (1 - f_t) * x_t ,   y_t = h_t
    The forget gate f_t = sigmoid(gate) is DATA-DEPENDENT when a 2nd input is provided (a *selective*
    state like Mamba/GRU); with a single input it uses a learned per-channel decay (an EMA / fixed
    linear-attention). CAUSAL by construction — h_t depends only on x_{<=t}, so no future can leak.
    Sequential over the token axis (slower than a parallel op), which is the honest cost of recurrence."""
    def __init__(self, node_id, d_model=768):
        super().__init__(node_id, "scan")
        self.d_model = d_model
        self.log_decay = nn.Parameter(torch.zeros(d_model))   # learned per-channel decay (1-input case)

    def expected_input_dim(self, input_index=0):
        return self.d_model

    def forward(self, inputs):
        x = inputs[0]
        B, T, D = x.shape
        if len(inputs) >= 2:
            f = torch.sigmoid(inputs[1])                      # data-dependent forget gate in (0,1) -> selective state
        else:
            f = torch.sigmoid(self.log_decay).view(1, 1, D).expand(B, T, D)   # learned decay -> EMA/linear-attention
        i = 1.0 - f
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):                                    # causal sequential scan
            h = f[:, t] * h + i[:, t] * x[:, t]
            outs.append(h)
        return torch.stack(outs, dim=1)                       # [B, T, D]

