import torch
from rwnn.graph import RWNNGraph
from rwnn.mutator import get_gpt2_dag

def verify_gpt2_124m():
    print("=== Phase 4: GPT-2 (124M) Verification ===")
    
    # 1. Configuration matching GPT-2 (124M)
    vocab_size = 50257
    max_seq_len = 1024
    d_model = 768
    batch_size = 2
    seq_len = 256 # Reduced sequence length just for fast forward-pass testing

    # 2. Get exact GPT-2 DAG
    nodes, edges = get_gpt2_dag('gpt2', vocab_size, max_seq_len)
    print(f"Constructed 12-layer GPT-2 DAG config with {len(nodes)} nodes and {len(edges)} edges.")

    # 3. Compile under RWNNGraph
    print("Compiling GPT-2 RWNNGraph...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RWNNGraph(nodes, edges, global_d_model=d_model)
    model.to(device)
    print(f"Compilation successful on device: {device}!")

    # 4. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 5. Run forward and backward passes
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    print(f"Input tensor shape: {x.shape}")

    print("Running forward pass...")
    logits = model(x)
    print(f"Output logits shape: {logits.shape}")

    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected output shape {expected_shape}, but got {logits.shape}"
    print("✓ Forward pass successful!")

    print("Running backward pass...")
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
    loss.backward()
    print("✓ Backward pass successful!")
    print("=== GPT-2 (124M) H-DAG verified successfully! ===")

if __name__ == "__main__":
    verify_gpt2_124m()
