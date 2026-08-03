# Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models

This repository contains the official, publication-ready implementation and documentation for **Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models**, targeted for *Neural Information Processing Systems (NeurIPS 2026)*.

We introduce a framework where decoder-only language models are modeled as **Heterogeneous Directed Acyclic Graphs (H-DAGs)**, mapping continuous Euclidean coordinates $x \in [0, 1]^{65}$ directly into topologically compiled, GPU-parallelized sequence-mixing networks. The optimization space is searched by a stateful, autonomous LLM agent (Metis-Agent) that writes its own self-diagnostic Python code (SVD, Ridge regression) to propose candidate vectors.

To bypass the cold-start training overhead, we introduce **Lamarckian Weight Inheritance via continuous Nearest-Neighbor Ancestry Mapping**, allowing offspring models to instantly copy pre-trained weight tensors from their closest Pareto-front parents.

---

## 📂 Project Structure

```text
rwnn-llm/
├── README.md                       # This experiment replication guide
├── blog4.md                        # Formal NeurIPS-style academic paper draft
├── evolve_agentic.py               # Main Autonomous Agentic Search & Lamarckian training loop
├── calculate_agentic_hypervolume.py# Script to compute exact hypervolumes and plot convergence
├── generate_graph_visualization.py # Script to compile and generate NetworkX layout PNGs of H-DAGs
├── generate_all_elites_samples.py  # Script to run autoregressive generation/sampling of elite models
├── rwnn/                           # Core modular H-DAG compiler
│   ├── __init__.py
│   ├── nodes.py                    # Graph primitives (modular and atomic layers)
│   ├── graph.py                    # JIT-topological compiler, dimension alignment, and executor
│   └── mutator.py                  # Graph mutations and cycle checking
├── train.bin                       # Preprocessed Salesforce WikiText-2 training tokens (BPE)
├── val.bin                         # Preprocessed Salesforce WikiText-2 validation tokens (BPE)
└── assets/                         # Folder containing generated plots and graph layouts
```

---

## ⚡ Hardware & Memory Requirements

This scaled-up configuration is heavily optimized to run on an **NVIDIA GeForce RTX 4070 Ti (12GB VRAM)** or similar consumer-grade GPU:
- **`gpt2` (162.5M parameters)**: Peak VRAM: **`4.96 GB`**
- **`gpt2-medium` (247.6M parameters)**: Peak VRAM: **`7.80 GB`**
- **`gpt2-large` (290.1M parameters)**: Peak VRAM: **`9.32 GB`**

All runs employ a batch size of `8`, sequence length of `256`, and are compiled using PyTorch's `BFloat16` mixed-precision tracking to guarantee stability under a 12GB memory budget.

---

## 🚀 Recreating the Experiment

Follow these steps to reproduce the 45-generation scaled-up agentic search:

### 1. Environment Setup
Activate the dedicated conda environment loaded with pre-configured CUDA-12, PyTorch, tiktoken, and google-genai libraries:
```bash
conda activate RWNNLMM
```

Ensure your Google Gemini API key is configured inside a local `.env` file in the project root:
```text
GOOGLE_API_KEY=AIzaSy...
```

### 2. Run the Autonomous Agentic Search
Start the main optimization script. This will automatically clear any legacy directories, compile the BPE-tokenized datasets, initialize the initial population (loaded with GPT-2, GPT-2 Medium, GPT-2 Large, and Gated Sparse configurations), and run the Metis-Agent loop:
```bash
python evolve_agentic.py
```
*Tip: To run this in the background as a headless persistent process, use:*
```bash
nohup python -u evolve_agentic.py > agentic_evolution.log 2>&1 &
```

During this search:
- Candidates ranging from 100M to 500M parameters are trained on WikiText-2 for 1,000 steps.
- Elites matched via continuous ancestry are promoted to subsequent generations to continue their training.
- Offspring inherit parent parameters in-place via `.copy_()` if they match the continuous distance neighborhood ($<0.6$).

### 3. Compute Hypervolume and Plot Convergence
After several generations of search have completed, you can calculate the exact mathematical hypervolume (S-Metric) dominated by the Pareto-front elites and generate beautiful, publication-ready convergence plots:
```bash
python calculate_agentic_hypervolume.py
```
This script dynamically computes the bounding boxes of all historical and current Pareto-front members, outputs statistics on-screen, and saves two high-resolution plots under `assets/`:
- `assets/agentic_hypervolume_progression.png`
- `assets/agentic_loss_progression.png`

### 4. Generate NetworkX Graph Visualizations
To compile and visualize the exact topological wiring and multi-hop skip residuals of the non-dominated elite architectures:
```bash
python generate_graph_visualization.py
```
This will output high-resolution NetworkX graph layout maps under the `assets/` directory.

### 5. Generate Autoregressive Appendix Samples
To sample language completions from the pre-trained elite weights of the final Pareto front and replicate the Appendix of the paper:
```bash
python generate_all_elites_samples.py
```
This runs the autoregressive causal generator on the Roman Empire prompt, producing fluent completed passages and saving them in text files.

---

## 📄 Academic Citation
If you utilize this H-DAG compiler, Lamarckian Weight Inheritance, or Agentic Optimizer framework in your research, please cite our draft:
```bibtex
@inproceedings{kyriacou2026lamarckian,
  title={Lamarckian Weight Inheritance in Autonomous H-DAG Large Language Models},
  author={Kyriacou, Stylianos},
  booktitle={Neural Information Processing Systems (NeurIPS 2026)},
  year={2026}
}
```