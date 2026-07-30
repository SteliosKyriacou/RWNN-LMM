import torch
import torch.nn as nn
from rwnn.graph import RWNNGraph

def test_atomic_layernorm():
    print("=== Phase 5: Numeric Verification of Atomic LayerNorm Sub-Graph ===")
    
    # 1. Configuration
    d_model = 128
    batch_size = 4
    seq_len = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Construct LayerNorm completely out of Atomic Nodes
    # We will implement: LayerNorm(x) = ((x - mean(x)) / sqrt(var(x) + eps)) * gamma + beta
    # Inside RWNNGraph:
    # Node 0: input placeholder (hidden state x)
    # Node 1: mean_reduce (dim=-1) -> outputs mean(x)
    # Node 2: subtract (inputs: [Node 0, Node 1]) -> outputs x - mean(x)
    # Node 3: square (input: Node 2) -> outputs (x - mean(x))^2
    # Node 4: mean_reduce (dim=-1) -> outputs var(x) = mean((x - mean(x))^2)
    # Node 5: sqrt (input: Node 4, eps=1e-5) -> outputs sqrt(var(x) + eps)
    # Node 6: divide (inputs: [Node 2, Node 5]) -> outputs normalized tensor
    # Node 7: scale_shift (input: Node 6, d_model=128) -> outputs scaled & shifted tensor
    
    nodes_config = [
        {'id': 0, 'type': 'input', 'kwargs': {}},
        {'id': 1, 'type': 'mean_reduce', 'kwargs': {'dim': -1}},
        {'id': 2, 'type': 'subtract', 'kwargs': {}},
        {'id': 3, 'type': 'square', 'kwargs': {}},
        {'id': 4, 'type': 'mean_reduce', 'kwargs': {'dim': -1}},
        {'id': 5, 'type': 'sqrt', 'kwargs': {'eps': 1e-5}},
        {'id': 6, 'type': 'divide', 'kwargs': {}},
        {'id': 7, 'type': 'scale_shift', 'kwargs': {'d_model': d_model}}
    ]

    edges_config = [
        (0, 1), # x -> mean(x)
        (0, 2), # x -> subtract (input 0)
        (1, 2), # mean(x) -> subtract (input 1)
        (2, 3), # (x-mean) -> square
        (3, 4), # (x-mean)^2 -> mean_reduce -> var(x)
        (4, 5), # var(x) -> sqrt -> std(x)
        (2, 6), # (x-mean) -> divide (input 0)
        (5, 6), # std(x) -> divide (input 1)
        (6, 7)  # normalized -> scale_shift
    ]

    # Compile the Atomic sub-graph
    print("Compiling Atomic LayerNorm Sub-Graph...")
    atomic_model = RWNNGraph(nodes_config, edges_config, global_d_model=d_model)
    atomic_model.to(device)
    print("Compilation successful!")

    # 3. Instantiate Native PyTorch LayerNorm for direct comparison
    native_ln = nn.LayerNorm(d_model, eps=1e-5).to(device)
    
    # Clone weights and biases to match exactly
    with torch.no_grad():
        atomic_model.nodes['7'].gamma.copy_(native_ln.weight)
        atomic_model.nodes['7'].beta.copy_(native_ln.bias)

    # 4. Prepare Inputs
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # 5. Forward Pass Numeric Comparison
    print("Comparing forward pass outputs...")
    with torch.no_grad():
        out_atomic = atomic_model(x)
        out_native = native_ln(x)

    # Calculate absolute difference
    max_diff = torch.max(torch.abs(out_atomic - out_native)).item()
    print(f"Max Absolute Numeric Difference: {max_diff:.3e}")
    
    assert max_diff < 1e-5, f"Numeric discrepancy too high! Diff: {max_diff:.3e}"
    print("✓ Numeric Equivalence verification PASSED!")

    # 6. Backward Pass Numeric Comparison (gradients flow)
    print("Comparing backward pass gradients...")
    atomic_model.zero_grad()
    native_ln.zero_grad()

    out_atomic = atomic_model(x)
    out_native = native_ln(x)

    loss_atomic = out_atomic.sum()
    loss_native = out_native.sum()

    loss_atomic.backward()
    loss_native.backward()

    # Compare gradients of gamma and beta
    grad_gamma_atomic = atomic_model.nodes['7'].gamma.grad
    grad_gamma_native = native_ln.weight.grad
    max_grad_diff = torch.max(torch.abs(grad_gamma_atomic - grad_gamma_native)).item()
    print(f"Max Gradient Numeric Difference (gamma): {max_grad_diff:.3e}")

    assert max_grad_diff < 1e-4, f"Gradient discrepancy too high! Diff: {max_grad_diff:.3e}"
    print("✓ Gradient Equivalence verification PASSED!")
    print("=== Atomic LayerNorm numeric verification fully successful! ===")

if __name__ == "__main__":
    test_atomic_layernorm()
