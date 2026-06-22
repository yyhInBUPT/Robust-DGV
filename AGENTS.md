# Project instructions for Codex

This project is developed locally on macOS but executed on a remote Linux server.

## Important workflow

- Do not run heavy training or certification commands locally.
- Do not try to install PyTorch, torch_geometric, torch_sparse, or torch_scatter locally.
- Do not repeatedly run Python checks if dependencies are missing.
- Local work is for code editing, static inspection, and small syntax-only checks.
- Official experiments are run on the remote server `Baitaiyangshen`.

## Python command

On this Mac, use `python3` instead of `python`.

If a command requires unavailable dependencies such as torch, torch_geometric, torch_sparse, or torch_scatter, do not keep retrying locally. Instead, report the exact command that should be run on the server.

## Server workflow

After editing code, the user will sync to server with:

```bash
./tools/sync_to_server.sh
```

Server run path:

```bash
cd ~/Robust_deepGNN/robust_grn
conda activate dgv
PYTHONPATH=..:. python train_deep.py ...
```

## **Editing rules**

- Make small, targeted changes.
- Do not refactor unrelated files.
- Do not modify DGV mathematical logic unless explicitly requested.
- For RCAEval work, prefer changing:
  - scripts/prepare_rcaeval_ob.py
  - robust_grn/utils.py
  - robust_grn/train_deep.py
  - robust_grn/certify.py
- Always summarize changed files and intended server-side test commands.