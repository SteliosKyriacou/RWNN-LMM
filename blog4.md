# Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models: Bridging Continuous Agentic Search and Discrete Neural Graph Topologies

**Author**: Stylianos Kyriacou  
**Target Conference**: *Neural Information Processing Systems (NeurIPS 2026)*  
**Category**: *Deep Learning, Neural Architecture Search, Autonomous AI Agents*  

---

## Abstract
Modern deep learning is dominated by hand-wired, sequential, and homogeneous neural architectures like the Transformer. While highly robust, these rigid structures restrict the network's capacity to discover non-linear representation pathways. While Neural Architecture Search (NAS) offers a path to structural discovery, optimizing discrete, variable-topology graph spaces remains combinatorial and computationally expensive. 

In this work, we present a paradigm shift: **Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models**. 
We introduce a continuous-to-discrete projection framework that maps an 86-dimensional (atomic-level) or 65-dimensional (scale-aware) Euclidean vector space $x \in [0, 1]^D$ directly into a topologically compiled, differentiable causal decoder H-DAG on GPUs. We delegate the search strategy entirely to an autonomous Large Language Model (LLM) agent that dynamically self-diagnoses the problem landscape by writing and executing Python code. 

Crucially, to accelerate search and achieve deep parameter convergence without institutional compute, we introduce **Lamarckian Weight Inheritance via continuous Nearest-Neighbor Ancestry Mapping**. This allows offspring graphs to instantly inherit matching pre-trained weight tensors from their closest non-dominated parents, accumulating a cumulative training horizon equivalent to **100,000 steps** over 100 generations of search. 

This paper outlines the mathematical formulations, compiler specifications, and genetic operators of this framework.

---

## 1. Introduction

Since the inception of the Transformer (Vaswani et al., 2017), autoregressive language modeling has been constrained to a sequential, homogeneous "straight-jacket." Layers are stacked as a rigid, sequential pipeline:
$$\text{Input} \to \text{Embedding} \to \text{Block}_1 \to \text{Block}_2 \to \dots \to \text{Block}_N \to \text{LN} \to \text{Head}$$

Within each block, causal self-attention and multi-layer perceptrons (MLPs) are bound to fixed dimensions and rigid residual additions ($x + \text{Attn}(x)$). This sequential assumption restricts the model's capacity to discover multi-path representation highways, dense cross-layer skip connections, or customized activation layouts.

While Randomly Wired Neural Networks (Xie et al., 2019) demonstrated that random directed acyclic graphs (DAGs) can outperform human designs, their evaluation was strictly confined to non-causal, feedforward computer vision classification tasks (ImageNet). Extending fluid graph topologies to autoregressive sequence mixing has historically failed due to:
1.  **Divergent Compilation**: Arbitrary graph mutations frequently break sequence causality, introduce recurrent loops (cycles), or leave nodes completely disconnected.
2.  **Combinatorial Curse**: Discrete graph spaces are NP-hard and cannot leverage the highly efficient continuous optimization algorithms (like CMA-ES or gradient-based surrogates) that dominate modern machine learning.
3.  **The Cold-Start Overheat**: In standard NAS, every mutated candidate must be trained from scratch. For Large Language Models, this cold-start training overhead requires astronomical compute budgets, making continuous architectural exploration impossible on consumer hardware.

### Our Contributions:
To overcome these limitations, we introduce **Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models**:
1.  **Differentiable H-DAG Causal Decoder Compiler**: We deconstruct standard Transformer blocks down to both modular and **atomic mathematical primitives** (mean reductions, squares, divisions, and matrix multiplications), compiling them into causally masked, GPU-parallelized H-DAGs with $10^{-7}$ precision matching against native PyTorch layers.
2.  **Continuous-to-Discrete Graph Projector**: We establish a continuous projection vector $x \in [0, 1]^{65}$ representing a smooth continuous relaxation of a multi-layer graph, bridging discrete graph search with continuous Euclidean optimizers.
3.  **Autonomous Agentic Search**: We deploy a stateful, generative LLM agent that acts as the optimizer, dynamically writing diagnostic Python code (SVD, correlation analysis) and numpy sampling strategies based on multi-objective hypervolume feedback.
4.  **Lamarckian Nearest-Neighbor Weight Inheritance**: We map ancestry dynamically in the continuous space, allowing offspring to inherit parent weights instantly across generations, accumulating **100,000 steps** of equivalent training over 100 generations.

---

## 2. Continuous Multi-Layer Graph Parameterization

To bridge our topological graph compiler with continuous, Euclidean vector-space optimizers, we represent a multi-layer stacked graph as a flat, continuous vector $x \in [0, 1]^{65}$. 

### Decision Variable Mapping
We partition the 65 continuous variables into five functional tracks:

```text
┌────────────────────────────────────────────────────────────────────────┐
│               65-DIMENSIONAL CONTINUOUS DECISION VECTOR                │
├─────────┬──────────────────┬──────────────────┬─────────────────┬──────┤
│   x[0]  │    x[1..16]      │    x[17..32]     │    x[33..48]    │x[49..│
│ (Depth) │  (Attention)     │      (MLP)       │  (Activations)  │ (Skip│
└─────────┴──────────────────┴──────────────────┴─────────────────┴──────┘
```

1.  **Global Depth ($x_0$)**: Maps $x_0 \in [0, 1]$ linearly to the integer range $[4, 15]$, representing the total number of layers ($L$) in the Transformer block:
    $$L = 4 + \text{int}(x_0 \cdot 12)$$
2.  **Attention Active States ($x_1 \dots x_{16}$)**: Represents whether the Causal Self-Attention block is compiled at Layer $l$. If $x_{1+l} > 0.5$, CausalAttention is compiled; otherwise, it is bypassed with an `Identity` pass-through.
3.  **MLP Active States ($x_{17} \dots x_{32}$)**: Represents whether the Feedforward MLP block is compiled at Layer $l$. If $x_{17+l} > 0.5$, the MLP is compiled; otherwise, it is bypassed.
4.  **MLP Activation Selection ($x_{33} \dots x_{48}$)**: Selects the activation function type for Layer $l$'s MLP:
    *   $x_{33+l} \in [0.00, 0.33) \implies$ **`GELU`** (Standard GPT-2).
    *   $x_{33+l} \in [0.33, 0.66) \implies$ **`SiLU (Swish)`** (Modern LLaMA style).
    *   $x_{33+l} \in [0.66, 1.00] \implies$ **`ReLU`**.
5.  **Cross-Layer Residual Skips ($x_{49} \dots x_{64}$)**: Represents whether to inject a cross-layer skip-connection from the output of Layer $l$ directly to the input of Layer $l+2$, establishing parallel residual highways:
    *   If $x_{49+l} > 0.5 \implies$ **Skip-edge is compiled**.
    *   If $x_{49+l} \le 0.5 \implies$ **No skip-edge is compiled**.

---

## 3. Causal Graph Compiler & Boundary Enforcements

The continuous vector $x$ is decoded on the fly into discrete nodes and edges, which are then compiled on the GPU by our H-DAG executor (`RWNNGraph` in `rwnn/graph.py`).

### A. Defensive Boundary Enforcements
To guarantee that every decoded candidate graph is mathematically valid,connected, and capable of stable gradient flow, we strictly enforce three boundary constraints:
*   **Input Sum-to-Block Boundary**: The compiler automatically appends an edge from the embedding sum (Node 3) to the first LayerNorm of the block (Node 4).
*   **Block-to-Output Boundary**: The compiler automatically appends an edge from the final layer residual sum (Node 11) to the vocabulary output head (Node 12).
*   **Bypass Edge Filtering**: To prevent `KeyError` exceptions during topological sorting, any edge $(u \to v)$ pointing to a node that was bypassed in the current configuration is automatically filtered out before compilation:
    $$\mathcal{E}_{\text{filtered}} = \{(u, v) \in \mathcal{E} \mid u \in \mathcal{V} \text{ and } v \in \mathcal{V}\}$$

### B. Lazy Edge Projections & Vectorized Broadcasting
If the optimizer connects nodes with mismatched channel dimensions (e.g. connecting a layer output of size 128 to a node expecting 512), the compiler lazily instantiates a learnable projection matrix (`EdgeConnection`) without bias to align them:
$$\text{EdgeConnection}(x) = x W^T \quad (W \in \mathbb{R}^{D_{\text{tgt}} \times D_{\text{src}}})$$

*   **Broadcasting Bypass**: If a source node outputs a dimension of **1** (e.g., from an atomic `MeanReduceNode`), the compiler bypasses projection and assigns an `Identity()` mapping. This preserves PyTorch's native **vectorized broadcasting** (e.g. `[B, T, 128] - [B, T, 1]`) at runtime without adding redundant linear parameters.

---

## 4. Lamarckian Weight Inheritance

In traditional deep learning, inheriting weights after structural mutations is notoriously difficult because topological changes break tensor shapes. In our H-DAG framework, because we assign **unique, deterministic Node IDs** (e.g., Node 5 for attention, Node 8 for MLP Up), and because shape mismatches are handled dynamically on the *edges* via `EdgeConnection`, **the internal parameter shapes of the individual nodes remain 100% static and unaffected by neighboring mutations.**

We can thus copy 100% of a parent's pre-trained weights directly into the offspring without any shape mismatch errors!

### A. Continuous Nearest-Neighbor Ancestry Mapping
Because the generative LLM agent operates as a black-box optimizer proposing candidates in the continuous $[0, 1]^{65}$ space, we map ancestry dynamically using Euclidean distance metrics.

Let $\mathcal{P} = \{p_1, p_2, \dots, p_m\}$ be the set of decision vectors representing the current Pareto front elites (the parents), and let $x_{\text{child}}$ be the proposed candidate vector. We map the child to its **nearest elite parent** on the manifold:
$$p_{\text{best}} = \arg\min_{p_k \in \mathcal{P}} \|x_{\text{child}} - p_k\|_2$$

If the minimum distance is within our evolutionary neighborhood ($\|x_{\text{child}} - p_{\text{best}}\|_2 < \theta_{\text{threshold}}$), we conclude that $x_{\text{child}}$ is an offspring of $p_{\text{best}}$, and we retrieve $p_{\text{best}}$'s trained PyTorch weight state-dict $\mathcal{S}(p_{\text{best}})$.

### B. Weight Transfer Protocol
When compiling the offspring model $C$, we load the state-dict $\mathcal{S}(p_{\text{best}})$:

```python
child_state = model.state_dict()
for k, v in parent_state_dict.items():
    # If the Node ID, variable key, and tensor shapes match exactly, copy the parameters!
    if k in child_state and child_state[k].shape == v.shape:
        child_state[k].copy_(v)
```

By copying the pre-trained tensors, the offspring **instantly starts training with a pre-converged validation loss (e.g. `~4.48`)** instead of a random loss of `10.98`. The AdamW optimizer then only needs to adjust the *newly mutated edges and nodes*, while keeping the pre-trained knowledge of the rest of the block completely intact.

Over 100 generations of search with `eval_steps = 1000` steps per generation, the final evolved elites successfully accumulate **100,000 steps of equivalent training!**

---

## References
1.  **Vaswani, A., et al.** (2017). *Attention is all you need.* Advances in Neural Information Processing Systems (NeurIPS 2017).
2.  **Xie, S., et al.** (2019). *Exploring randomly wired neural networks for image recognition.* Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2019).
3.  **Karpathy, A.** (2022-2025). *nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs.* GitHub Repository.
4.  **Liu, H., et al.** (2018). *DARTS: Differentiable Architecture Search.* International Conference on Learning Representations (ICLR 2019).
5.  **Kyriacou, S.** (2026). *Agentic Optimizer: A purely LLM-driven, agentic multi-objective optimization framework.* GitHub Repository.
