#!/bin/bash
set -e

rsync -avz \
  --exclude ".git" \
  --exclude ".idea" \
  --exclude ".vscode" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  --exclude ".venv" \
  --exclude "venv" \
  --exclude "*.pt" \
  --exclude "robust_grn/dataset" \
  --exclude "图" \
  /Users/yuhang/CODE/Robust_deepGNN/ \
  Baitaiyangshen:~/Robust_deepGNN/