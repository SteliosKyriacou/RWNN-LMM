import torch
from rwnn.graph import RWNNGraph
from rwnn.mutator import get_canonical_nano_gpt

def verify_dag_pipeline():
    print("=== Phase 1: Verification Pipeline ===")
    
    # 1. Configuration
    vocab_size = 1000
    max_seq_len = 256
    d_model = 128
    batch_size = 4
    seq_len = 64

    # 2. Get canonical nanoGPT nodes and edges
    nodes, edges = get_canonical_nano_gpt(vocab_size, max_seq_len, d_model)
    print(f"Constructed canonical nanoGPT config with {len(nodes)} nodes and {len(edges)} edges.")

    # 3. Instantiate the RWNNGraph compiler
    print("Compiling RWNNGraph...")
    model = RWNNGraph(nodes, edges, global_d_model=d_model)
    print("Compilation successful!")

    # 4. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 5. Prepare dummy inputs
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Dummy inputs shape: {x.shape}")

    # 6. Execute forward pass
    print("Executing forward pass...")
    logits = model(x)
    print(f"Forward pass output shape: {logits.shape}")
    
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected output shape {expected_shape}, but got {logits.shape}"
    print("✓ Forward pass verification PASSED!")

    # 7. Execute backward pass
    print("Executing backward pass...")
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
    loss.backward()
    print("✓ Backward pass verification PASSED!")
    print("=== Verification Successful! ===")

if __name__ == "__main__":
    verify_dag_pipeline()
