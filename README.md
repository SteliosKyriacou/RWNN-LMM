# Randomly Wired Neural Networks as Large Language Models (RWNN-LLM)

## 🎯 Goal
The goal of this project is to implement, train, and optimize **Randomly Wired Neural Networks (RWNNs)** in the domain of **Large Language Models (LLMs)**.

Specifically, we design a framework where a decoder-only language model is modeled as a **Heterogeneous Directed Acyclic Graph (H-DAG)**. Within this graph:
- **Nodes** represent primitive modular tensor operations (e.g., Causal Self-Attention, Linear Projections, Layer Normalization, Activations, Element-wise Addition).
- **Edges** represent communication pathways where hidden states flow.

By representing an LLM as a modular H-DAG, we can completely discard human-designed, homogeneous sequential architectures (like the standard stack of Transformer blocks in `nanoGPT`) and use a **Memetic Algorithm (Multi-Objective Evolutionary Algorithm + Gradient-Based Backpropagation)** to randomly or optimally wire the network. This allows us to search the structural frontier and find high-efficiency, low-complexity models.

---

## 🏗️ Architectural Foundations

### 1. Heterogeneous Graph Nodes (H-DAG)
Unlike simple RWNNs which utilize identical scalar neurons, an LLM requires structured tensor manipulations. We define a library of primitive PyTorch-based nodes operating on sequences `[Batch, SeqLen, Channels]`:
- **Embedding Nodes**: `TokenEmbeddingNode`, `PositionalEmbeddingNode`
- **Parameterized Projections**: `LinearNode(d_in, d_out)`
- **Attention Modules**: `CausalAttentionNode(n_heads, d_model)`
- **Normalization & Non-linearities**: `LayerNormNode`, `ActivationNode(GELU/SiLU)`
- **Combinators**: `SumNode` (Residual addition), `ConcatNode`, `ElementMulNode` (Gating)

### 2. Lazy Edge Projections (Dimension Alignment)
In a randomly wired network, a mutation might add a connection between two nodes with mismatched channel dimensions. To make the architecture fully robust to arbitrary structural mutations, we wrap every edge in an `EdgeConnection` module:
- It checks the source and target dimensions dynamically.
- If dimensions match, it is a zero-cost pass-through.
- If dimensions mismatch, it lazily instantiates a learnable `nn.Linear` projection without bias, ensuring shape consistency.

### 3. GPU-Parallelized Topological Execution
To resolve the sequential bottleneck of arbitrary graphs, we compile the H-DAG into **topological levels**:
- All nodes in a given level $L_r$ are computationally independent.
- We evaluate all nodes within $L_r$ in parallel using batched matrix multiplications, minimizing kernel launches and leveraging GPU vectorization.

---

## 📈 Memetic & Multi-Objective Optimization Loop

We employ a dual optimization scheme:
1. **Inner Loop (Gradient Descent)**: Fine-tune node weights and edge projections using standard sequence forecasting cross-entropy loss (Backprop).
2. **Outer Loop (Evolutionary Search)**: Mutate the graph structure to discover optimal layouts.

```
       Cross-Entropy (Loss)
         ▲
         │  * (Inefficient Random Wirings)
         │       *
         │             * (Evolved High-Performance Configurations)
         │  ───────┐
         │         └───* (Pareto Front of Optimally Wired Networks)
         │             └───────*
         └──────────────────────────────► Complexity (FLOPs/Synapses)
```

### Mutation Primitives:
- **Add Node**: Splices a new node on an existing edge.
- **Remove Node**: Bypasses an existing node, merging incoming and outgoing paths.
- **Add Edge**: Creates a new tensor flow pathway (uses DFS cycle-detection to prevent recurrence).
- **Remove Edge**: Deletes a pathway, ensuring graph reachability is preserved.

---

## ⚙️ Project Structure
```text
rwnn-llm/
├── README.md               # This project documentation
├── rwnn/
│   ├── __init__.py
│   ├── nodes.py            # Primitive H-DAG Nodes (Attention, Linear, Norm, etc.)
│   ├── graph.py            # RWNNGraph compiler and topological execution engine
│   └── mutator.py          # Evolutionary operations (Mutation, Crossover, Cycle Checks)
├── train.py                # Character-level training pipeline on Tiny Shakespeare
└── evolve.py               # Multi-objective Memetic Optimization engine
```
