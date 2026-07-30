import os
import sys
import time
import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from rwnn.graph import RWNNGraph
from rwnn.mutator import get_canonical_nano_gpt, get_gpt2_124m_dag

# 1. Dataset Downloading
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"

def download_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading Tiny Shakespeare dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")

# Character-level Tokenizer (for fast/fun character training)
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.char2idx[c] for c in s]

    def decode(self, l):
        return ''.join([self.idx2char[i] for i in l])


def train_rwnn_nanogpt():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='toy', choices=['toy', 'gpt2'], 
                        help="toy = 6-layer (nanoGPT style), gpt2 = 12-layer 124M GPT-2")
    parser.add_argument('--vocab_type', type=str, default='char', choices=['char', 'gpt2_bpe'],
                        help="char = character level tokenizer, gpt2_bpe = official GPT-2 tokenizer")
    parser.add_argument('--max_iters', type=int, default=1000, help="number of training iterations")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--block_size', type=int, default=256, help="context window size (block size)")
    parser.add_argument('--lr', type=float, default=6e-4, help="max learning rate")
    parser.add_argument('--warmup_iters', type=int, default=100, help="lr warmup iterations")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="weight decay")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="gradient clip threshold")
    parser.add_argument('--eval_interval', type=int, default=100, help="evaluate model every N steps")
    parser.add_argument('--sample_interval', type=int, default=200, help="sample generation every N steps")
    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device.upper()}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for Tensor Cores on Ampere/Ada Lovelace GPUs like RTX 4070 Ti
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. Data Preparation
    download_data()
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    # Configure Tokenizer
    if args.vocab_type == 'char':
        tokenizer = CharTokenizer(text)
        vocab_size = tokenizer.vocab_size
        print(f"Using Character-level Tokenizer. Vocab size: {vocab_size}")
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    else:
        print("Loading official GPT-2 BPE Tokenizer from HuggingFace...")
        bpe_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        vocab_size = bpe_tokenizer.vocab_size
        print(f"Using GPT-2 BPE Tokenizer. Vocab size: {vocab_size}")
        data = torch.tensor(bpe_tokenizer.encode(text), dtype=torch.long)

    # Train / Val Split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i:i+args.block_size] for i in ix])
        y = torch.stack([d[i+1:i+args.block_size+1] for i in ix])
        return x.to(device), y.to(device)

    # 2. Model Architecture Definitions
    if args.model_type == 'gpt2':
        d_model = 768
        n_layer = 12
        print("Constructing 12-layer, 768-dim, 12-head GPT-2 (124M) equivalent DAG...")
        nodes, edges = get_gpt2_124m_dag(vocab_size, args.block_size, d_model, n_layer)
    else:
        # toy: nanoGPT style toy model for fast training on character levels
        d_model = 384
        n_layer = 6
        print("Constructing 6-layer, 384-dim, 6-head nanoGPT style Toy DAG...")
        # Stack 6 canonical block structures
        nodes, edges = get_gpt2_124m_dag(vocab_size, args.block_size, d_model, n_layer)

    # Compile H-DAG
    print("Compiling RWNNGraph...")
    model = RWNNGraph(nodes, edges, global_d_model=d_model)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model compiled successfully. Total parameters: {total_params:,}")

    # 3. Optimizer with Weight Decay (excluding biases and norms as in nanoGPT)
    param_dict = {pn: p for p, pn in zip(model.parameters(), model.state_dict().keys()) if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))

    # Cosine learning rate decay with linear warmup (nanoGPT style)
    def get_lr(it):
        if it < args.warmup_iters:
            return args.lr * it / args.warmup_iters
        if it > args.max_iters:
            return args.lr * 0.1
        decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.lr * (0.1 + 0.9 * coeff)

    # Evaluation loss estimator
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(10)
            for k in range(10):
                X, Y = get_batch(split)
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(X)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # Sampling helper
    def sample(model, max_new_tokens=150):
        model.eval()
        if args.vocab_type == 'char':
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            context = torch.tensor([[bpe_tokenizer.bos_token_id or 50256]], dtype=torch.long, device=device)
        
        generated = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context_cond = context if context.size(1) <= args.block_size else context[:, -args.block_size:]
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(context_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, next_token), dim=1)
                generated.append(next_token.item())
        
        model.train()
        if args.vocab_type == 'char':
            return tokenizer.decode(generated)
        else:
            return bpe_tokenizer.decode(generated)

    # 4. Training Loop
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    print("\n=== Beginning Training Execution ===")
    
    convergence_history = []
    
    t0 = time.time()
    for it in range(args.max_iters):
        # Update learning rate
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate model periodically
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss()
            print(f"Iter {it:4d}: Train Loss = {losses['train']:.4f}, Val Loss = {losses['val']:.4f} | LR = {lr:.6f}")
            convergence_history.append({
                'iter': it,
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'lr': lr
            })

        # Sample generation periodically
        if it > 0 and it % args.sample_interval == 0:
            print("\n" + "="*40 + f" Sample at Iter {it} " + "="*40)
            print(sample(model, max_new_tokens=150))
            print("=" * 100 + "\n")

        # Get batch
        xb, yb = get_batch('train')

        # Forward pass under AMP mixed-precision (BF16/FP16)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

        # Backward and step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()

        # Print throughput logging
        if it % 10 == 0 and it > 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_processed = 10 * args.batch_size * args.block_size
            tok_per_sec = tokens_processed / dt
            print(f"Step {it:4d} | Loss = {loss.item():.4f} | {dt*1000/10:.1f}ms/step | {tok_per_sec:.1f} tokens/sec")

    print("\nTraining complete! Generating final sample:")
    print("=" * 100)
    print(sample(model, max_new_tokens=300))
    print("=" * 100)

    import json
    with open("convergence.json", "w") as f:
        json.dump(convergence_history, f, indent=4)
    print("Convergence history saved to convergence.json")


if __name__ == "__main__":
    train_rwnn_nanogpt()
