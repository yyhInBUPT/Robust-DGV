# Robust-DGV

Code and experiment scripts for certified robustness analysis of Deep Graph Neural Networks with DGV.

## Overview

This repository contains the core implementation of robust certification for graph neural networks, including GCNII-based models, training scripts, certification scripts, RCAEval data preparation utilities, and ablation-study runners used for the revision experiments.

Large artifacts are intentionally not included in the repository:

- trained checkpoints
- generated results and figures
- processed datasets
- local IDE/cache files

The `.gitignore` file excludes these artifacts by default.

## Repository Structure

```text
robust_grn/
  model.py                    # GCNII and robust certification model
  train_deep.py               # training entry point
  certify.py                  # certification entry point
  utils.py                    # dataset and graph utilities
  figure8_degree_maxq.py      # degree/Max-Q experiment script
  rcaeval_degree_maxq.py      # RCAEval degree/Max-Q analysis

scripts/
  run_ablation_studies.py     # unified ablation runner
  prepare_rcaeval_ob.py       # RCAEval Online Boutique preprocessing
  prepare_rcaeval_re1.py      # RCAEval Train Ticket preprocessing
  summarize_rcaeval_cert.py   # certification CSV summarization
  make_rcaeval_layer_table.py # layer-ablation table utility

tools/
  sync_to_server.sh           # optional local-to-server sync helper
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric should be installed with versions compatible with your CUDA driver. If the generic installation fails, follow the official PyTorch and PyG installation instructions for your platform.

## Data and Checkpoints

Place datasets and checkpoints under the paths expected by the scripts:

```text
robust_grn/dataset/
robust_grn/pretrained/
```

These directories are ignored by Git because they can be large and environment-specific.

## Basic Usage

Train a GCNII model:

```bash
cd robust_grn
PYTHONPATH=..:. python3 train_deep.py --data cora --layer 16
```

Run certification:

```bash
cd robust_grn
PYTHONPATH=..:. python3 certify.py --data cora --layer 16 --q 14 --Q-list 1 10 --split test --output result/cora_cert.csv
```

Run all ablation studies:

```bash
PYTHONPATH=..:. python3 scripts/run_ablation_studies.py --output-dir result/ablation --task all
```

Dry-run the ablation grid without training or certification:

```bash
PYTHONPATH=..:. python3 scripts/run_ablation_studies.py --output-dir result/ablation --task all --dry-run
```

## Ablation Outputs

The unified ablation script writes stable CSV/figure/summary outputs:

```text
result/ablation/talpha_all_datasets.csv
result/ablation/q_sensitivity.csv
result/ablation/q_sensitivity.png
result/ablation/depth_sensitivity.csv
result/ablation/depth_sensitivity.png
result/ablation/gcnii_hyper_sensitivity.csv
result/ablation/ablation_summary.md
```

Each experiment configuration is also saved as JSON under:

```text
result/ablation/configs/
```

## Notes

- The scripts are designed for Linux servers with CUDA-capable GPUs.
- Full certification can be computationally expensive on large datasets.
- Generated results, checkpoints, and datasets should be archived separately from Git.

