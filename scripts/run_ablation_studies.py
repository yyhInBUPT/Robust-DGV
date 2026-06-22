#!/usr/bin/env python3
"""Run GCNII ablation studies for the TSC revision.

This script is intentionally a thin orchestration layer around the existing
robust_grn data loaders, RobustGRNModel, training loop style, checkpoints, and
certification routine.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBUST_GRN_DIR = REPO_ROOT / "robust_grn"
for path in (str(REPO_ROOT), str(ROBUST_GRN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F
from tqdm import tqdm

from robust_grn.model import RobustGRNModel
from robust_grn.utils import (
    accuracy,
    load_citation,
    normalized_adj_tensor,
    sparse_mx_to_torch_sparse_tensor,
    sys_normalized_adjacency,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "cora": {"display": "Cora", "loader": lambda: load_citation("cora"), "default_layers": 16},
    "citeseer": {"display": "Citeseer", "loader": lambda: load_citation("citeseer"), "default_layers": 32},
    "pubmed": {"display": "Pubmed", "loader": lambda: load_citation("pubmed"), "default_layers": 16},
    "actor": {"display": "Actor", "loader": lambda: process_actor_local(), "default_layers": 8},
    "amazon_cs": {"display": "Amazon Computers", "loader": lambda: process_amazon_npz("dataset/other/amazon_cs.npz"), "default_layers": 8},
    "amazon_photo": {"display": "Amazon Photo", "loader": lambda: process_amazon_npz("dataset/other/amazon_electronics_photo.npz"), "default_layers": 8},
}

PREFERRED_CHECKPOINTS = {
    ("cora", 16, 0.1, 0.5): "pretrained/48ccee97545b45f29c39161702dfe2d9.pt",
    ("citeseer", 32, 0.1, 0.5): "pretrained/cd35a8a4fba44d6cb28ea899171d7497.pt",
    ("pubmed", 16, 0.1, 0.5): "pretrained/223fc6a516b74fa2a8bf3d7597124f8e.pt",
}

FEATURE_DIM_FALLBACK = {
    "cora": 1433,
    "citeseer": 3703,
    "pubmed": 500,
    "actor": 932,
    "amazon_cs": 767,
    "amazon_photo": 745,
}

TASK_OUTPUTS = {
    "talpha": "talpha_all_datasets.csv",
    "q": "q_sensitivity.csv",
    "depth": "depth_sensitivity.csv",
    "gcnii_hyper": "gcnii_hyper_sensitivity.csv",
}


def load_npz(path):
    with np.load(path, allow_pickle=True) as loader:
        data = dict(loader)
    for key, value in data.items():
        if getattr(value, "dtype", None) is not None and value.dtype.kind in {"O", "U"}:
            data[key] = value.tolist()
    return data


def dense_features_from_npz(dataset):
    if "node_attr" in dataset:
        features = dataset["node_attr"].tocoo()
        return torch.sparse_coo_tensor(
            torch.LongTensor(np.vstack([features.row, features.col])),
            torch.FloatTensor(features.data),
            torch.Size(features.shape),
        ).to_dense()
    attr_indptr = torch.tensor(dataset["attr_indptr"], dtype=torch.int64)
    attr_indices = torch.tensor(dataset["attr_indices"], dtype=torch.int64)
    attr_data = torch.tensor(dataset["attr_data"], dtype=torch.float32)
    num_nodes = len(attr_indptr) - 1
    num_features = int(attr_indices.max()) + 1
    row_indices = torch.repeat_interleave(torch.arange(num_nodes), attr_indptr[1:] - attr_indptr[:-1])
    return torch.sparse_coo_tensor(
        torch.stack([row_indices, attr_indices]),
        attr_data,
        size=(num_nodes, num_features),
    ).to_dense()


def adjacency_from_npz(dataset):
    if "adj_matrix" in dataset:
        adj = dataset["adj_matrix"]
        dense = torch.sparse_coo_tensor(
            torch.LongTensor(np.vstack([adj.tocoo().row, adj.tocoo().col])),
            torch.FloatTensor(adj.tocoo().data),
            torch.Size(adj.shape),
        ).to_dense()
        return adj, dense
    adj_indptr = torch.tensor(dataset["adj_indptr"], dtype=torch.int64)
    adj_indices = torch.tensor(dataset["adj_indices"], dtype=torch.int64)
    adj_data = torch.tensor(dataset["adj_data"], dtype=torch.float32)
    num_nodes = len(adj_indptr) - 1
    row_indices = torch.repeat_interleave(torch.arange(num_nodes), adj_indptr[1:] - adj_indptr[:-1])
    dense = torch.sparse_coo_tensor(
        torch.stack([row_indices, adj_indices]),
        adj_data,
        size=(num_nodes, num_nodes),
    ).to_dense()
    return csr_matrix(dense.cpu().numpy()), dense


def labels_from_npz(dataset):
    if "node_label" in dataset:
        return torch.tensor(dataset["node_label"], dtype=torch.int64)
    return torch.tensor(dataset["labels"], dtype=torch.int64)


def random_split(num_nodes):
    indices = np.random.permutation(num_nodes)
    train_end = int(num_nodes * 0.7)
    val_end = int(num_nodes * 0.9)
    return (
        torch.tensor(indices[:train_end], dtype=torch.int64),
        torch.tensor(indices[train_end:val_end], dtype=torch.int64),
        torch.tensor(indices[val_end:], dtype=torch.int64),
    )


def process_amazon_npz(path):
    dataset = load_npz(path)
    features = dense_features_from_npz(dataset)
    labels = labels_from_npz(dataset)
    adj_sparse, dense = adjacency_from_npz(dataset)
    adj_sparse = adj_sparse + adj_sparse.T.multiply(adj_sparse.T > adj_sparse) - adj_sparse.multiply(adj_sparse.T > adj_sparse)
    adj = sparse_mx_to_torch_sparse_tensor(sys_normalized_adjacency(adj_sparse))
    dense = dense + dense.t() - torch.diag(dense.diagonal())
    dense = normalized_adj_tensor(dense)
    idx_train, idx_val, idx_test = random_split(features.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test, dense


def process_actor_local():
    from torch_geometric.datasets import Actor

    data = Actor(root="./dataset")[0]
    idx_test = torch.nonzero(data.test_mask[:, 0]).squeeze().long()
    idx_train = torch.nonzero(data.train_mask[:, 0]).squeeze().long()
    idx_val = torch.nonzero(data.val_mask[:, 0]).squeeze().long()
    features = data.x.float()
    labels = data.y.long()
    edge_index = data.edge_index.cpu()
    num_nodes = int(data.num_nodes)
    values = torch.ones(edge_index.shape[1], dtype=torch.float32)
    raw_adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).to_dense()
    raw_adj = raw_adj + raw_adj.t() - torch.diag(raw_adj.diagonal())
    dense_adj = normalized_adj_tensor(raw_adj)
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    adj_sparse = csr_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
    adj_sparse = adj_sparse + adj_sparse.T.multiply(adj_sparse.T > adj_sparse) - adj_sparse.multiply(adj_sparse.T > adj_sparse)
    adj = sparse_mx_to_torch_sparse_tensor(sys_normalized_adjacency(adj_sparse))
    return adj, features, labels, idx_train, idx_val, idx_test, dense_adj

CSV_FIELDS = {
    "talpha": [
        "dataset",
        "model",
        "layers",
        "q",
        "Q",
        "T_alpha",
        "certified_coverage",
        "avg_lower_bound",
        "runtime_total",
        "runtime_per_node",
        "num_nodes_certified",
        "checkpoint_path",
        "failure_reason",
    ],
    "q": [
        "dataset",
        "model",
        "layers",
        "q_ratio",
        "q",
        "Q",
        "certified_coverage",
        "avg_lower_bound",
        "runtime_per_node",
        "failure_reason",
    ],
    "depth": [
        "dataset",
        "model",
        "layers",
        "q",
        "Q",
        "certified_coverage",
        "avg_lower_bound",
        "runtime_per_node",
        "accuracy",
        "failure_reason",
    ],
    "gcnii_hyper": [
        "dataset",
        "model",
        "layers",
        "alpha_l",
        "lambda",
        "q",
        "Q",
        "accuracy",
        "certified_coverage",
        "avg_lower_bound",
        "runtime_per_node",
        "failure_reason",
    ],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def resolve_output_dir(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def to_device(data_tuple):
    adj, features, labels, idx_train, idx_val, idx_test, dense_adj = data_tuple
    return (
        adj.to(DEVICE),
        features.float().to(DEVICE),
        labels.long().to(DEVICE),
        idx_train.long().to(DEVICE),
        idx_val.long().to(DEVICE),
        idx_test.long().to(DEVICE),
        dense_adj.float().to(DEVICE),
    )


def load_dataset(dataset_key, seed):
    set_seed(seed)
    return to_device(DATASETS[dataset_key]["loader"]())


def build_model(features, labels, dense_adj, layers, alpha_l, lamda, args):
    return RobustGRNModel(
        nfeat=features.shape[1],
        adj=dense_adj,
        nlayers=layers,
        dim=[args.hidden_dim],
        nclass=int(labels.max()) + 1,
        dropout=args.dropout,
        lamda=lamda,
        alpha=alpha_l,
        variant=False,
    ).to(DEVICE)


def checkpoint_for(dataset_key, layers, alpha_l, lamda, output_dir):
    preferred = PREFERRED_CHECKPOINTS.get((dataset_key, layers, alpha_l, lamda))
    if preferred and Path(preferred).exists():
        return Path(preferred)
    return output_dir / "checkpoints" / "{}_gcnii_l{}_alpha{}_lambda{}.pt".format(
        dataset_key,
        layers,
        str(alpha_l).replace(".", "p"),
        str(lamda).replace(".", "p"),
    )


def train_model(model, features, adj, labels, idx_train, idx_val, checkpoint_path, args):
    optimizer = torch.optim.Adam(
        [
            {"params": model.params1, "weight_decay": args.wd1},
            {"params": model.params2, "weight_decay": args.wd2},
        ],
        lr=args.lr,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_epoch = 0
    bad_counter = 0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        if loss_val.item() < best:
            best = loss_val.item()
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_counter += 1
        if (epoch + 1) % args.log_every == 0:
            print(
                "epoch={} train_loss={:.4f} val_loss={:.4f}".format(
                    epoch + 1,
                    loss_train.item(),
                    loss_val.item(),
                )
            )
        if bad_counter == args.patience:
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return {
        "loaded_existing": False,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best),
        "train_runtime_seconds": float(time.time() - start),
        "epochs_requested": int(args.epochs),
        "patience": int(args.patience),
        "lr": float(args.lr),
        "wd1": float(args.wd1),
        "wd2": float(args.wd2),
        "dropout": float(args.dropout),
    }


def load_or_train(model, features, adj, labels, idx_train, idx_val, checkpoint_path, args):
    if checkpoint_path.exists() and not args.retrain:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        return {"loaded_existing": True}
    return train_model(model, features, adj, labels, idx_train, idx_val, checkpoint_path, args)


def clean_accuracy(model, features, adj, labels, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        return float(accuracy(output[idx_test], labels[idx_test]).item())


def selected_nodes(idx_test, args):
    nodes = idx_test.detach().cpu()
    if args.max_nodes is not None:
        nodes = nodes[: args.max_nodes]
    return nodes


def certify_with_lower_bounds(model, features, nodes, q_local, q_global, certify_adj, batch_size):
    robust = np.zeros(len(nodes), dtype=bool)
    lower_bound_values = []
    chunks = [nodes[i : i + batch_size] for i in range(0, len(nodes), batch_size)]
    model.eval()
    for offset, chunk in enumerate(tqdm(chunks, desc="Q={}".format(q_global), leave=False)):
        chunk_device = chunk.to(features.device)
        with torch.no_grad():
            lb = model.dual_backward(features, chunk_device, q_local, q_global, certify_adj)
        lb_np = lb.detach().cpu().numpy()
        robust[offset * batch_size : offset * batch_size + len(chunk)] = (lb_np > 0).sum(1) == model.nclass - 1
        non_target = lb_np.shape[1] - 1
        if non_target > 0:
            lower_bound_values.extend(np.sort(lb_np, axis=1)[:, :non_target].reshape(-1).tolist())
        else:
            lower_bound_values.extend(lb_np.reshape(-1).tolist())
    return robust, np.asarray(lower_bound_values, dtype=np.float64)


def run_certification(spec, args, output_dir):
    dataset_key = spec["dataset_key"]
    layers = int(spec["layers"])
    alpha_l = float(spec.get("alpha_l", args.alpha))
    lamda = float(spec.get("lambda", args.lamda))
    q = int(spec["q"])
    q_global = int(spec["Q"])
    adj, features, labels, idx_train, idx_val, idx_test, dense_adj = load_dataset(dataset_key, args.seed)
    model = build_model(features, labels, dense_adj, layers, alpha_l, lamda, args)
    checkpoint_path = checkpoint_for(dataset_key, layers, alpha_l, lamda, output_dir)
    train_info = load_or_train(model, features, adj, labels, idx_train, idx_val, checkpoint_path, args)
    acc = clean_accuracy(model, features, adj, labels, idx_test)
    nodes = selected_nodes(idx_test, args)
    start = time.time()
    robust, lower_bounds = certify_with_lower_bounds(
        model,
        features,
        nodes,
        q,
        q_global,
        dense_adj,
        args.cert_batch_size,
    )
    runtime = time.time() - start
    num_nodes = int(len(nodes))
    return {
        "dataset": DATASETS[dataset_key]["display"],
        "model": "GCNII",
        "layers": layers,
        "q": q,
        "Q": q_global,
        "accuracy": acc,
        "certified_coverage": float(robust.sum() / num_nodes * 100 if num_nodes else 0.0),
        "avg_lower_bound": float(lower_bounds.mean() if lower_bounds.size else 0.0),
        "runtime_total": float(runtime),
        "runtime_per_node": float(runtime / num_nodes if num_nodes else 0.0),
        "num_nodes_certified": num_nodes,
        "checkpoint_path": str(checkpoint_path),
        "train_info": train_info,
        "feature_dim": int(features.shape[1]),
    }


def make_specs(task, args):
    q_global_values = args.Q_list
    specs = []
    if task == "talpha":
        for dataset_key in ["cora", "citeseer", "pubmed", "actor", "amazon_cs", "amazon_photo"]:
            feature_dim = feature_dim_for(dataset_key, args.seed, prefer_fallback=args.dry_run)
            layers = DATASETS[dataset_key]["default_layers"] if args.layers is None else args.layers
            q = max(1, int(math.floor(0.01 * feature_dim)))
            for q_global in q_global_values:
                for t_alpha in [0, 1, 2, 3]:
                    specs.append(
                        {
                            "task": task,
                            "dataset_key": dataset_key,
                            "layers": layers,
                            "q": q,
                            "Q": q_global,
                            "T_alpha": t_alpha,
                            "alpha_l": args.alpha,
                            "lambda": args.lamda,
                        }
                    )
    elif task == "q":
        for dataset_key in ["cora", "amazon_photo"]:
            feature_dim = feature_dim_for(dataset_key, args.seed, prefer_fallback=args.dry_run)
            layers = DATASETS[dataset_key]["default_layers"] if args.layers is None else args.layers
            for q_ratio in [0.005, 0.01, 0.02, 0.05]:
                q = max(1, int(math.floor(q_ratio * feature_dim)))
                for q_global in q_global_values:
                    specs.append(
                        {
                            "task": task,
                            "dataset_key": dataset_key,
                            "layers": layers,
                            "q_ratio": q_ratio,
                            "q": q,
                            "Q": q_global,
                            "T_alpha": 0,
                            "alpha_l": args.alpha,
                            "lambda": args.lamda,
                        }
                    )
    elif task == "depth":
        for dataset_key in ["cora", "amazon_photo"]:
            feature_dim = feature_dim_for(dataset_key, args.seed, prefer_fallback=args.dry_run)
            q = max(1, int(math.floor(0.01 * feature_dim)))
            for layers in [4, 8, 16, 32, 64]:
                for q_global in q_global_values:
                    specs.append(
                        {
                            "task": task,
                            "dataset_key": dataset_key,
                            "layers": layers,
                            "q": q,
                            "Q": q_global,
                            "T_alpha": 0,
                            "alpha_l": args.alpha,
                            "lambda": args.lamda,
                        }
                    )
    elif task == "gcnii_hyper":
        for dataset_key in ["cora", "amazon_photo"]:
            feature_dim = feature_dim_for(dataset_key, args.seed, prefer_fallback=args.dry_run)
            layers = DATASETS[dataset_key]["default_layers"] if args.layers is None else args.layers
            q = max(1, int(math.floor(0.01 * feature_dim)))
            for alpha_l in [0.05, 0.1, 0.2]:
                for lamda in [0.25, 0.5, 1.0]:
                    for q_global in q_global_values:
                        specs.append(
                            {
                                "task": task,
                                "dataset_key": dataset_key,
                                "layers": layers,
                                "q": q,
                                "Q": q_global,
                                "T_alpha": 0,
                                "alpha_l": alpha_l,
                                "lambda": lamda,
                            }
                        )
    else:
        raise ValueError("Unknown task: {}".format(task))
    return specs


_FEATURE_DIM_CACHE = {}


def feature_dim_for(dataset_key, seed, prefer_fallback=False):
    cache_key = (dataset_key, seed)
    if cache_key not in _FEATURE_DIM_CACHE:
        if prefer_fallback and dataset_key in FEATURE_DIM_FALLBACK:
            _FEATURE_DIM_CACHE[cache_key] = FEATURE_DIM_FALLBACK[dataset_key]
            return _FEATURE_DIM_CACHE[cache_key]
        try:
            _, features, _, _, _, _, _ = load_dataset(dataset_key, seed)
            _FEATURE_DIM_CACHE[cache_key] = int(features.shape[1])
        except Exception:
            if dataset_key not in FEATURE_DIM_FALLBACK:
                raise
            _FEATURE_DIM_CACHE[cache_key] = FEATURE_DIM_FALLBACK[dataset_key]
    return _FEATURE_DIM_CACHE[cache_key]


def row_from_result(task, spec, result=None, failure_reason=""):
    dataset_name = DATASETS[spec["dataset_key"]]["display"]
    base = {
        "dataset": dataset_name,
        "model": "GCNII",
        "layers": spec["layers"],
        "q": spec["q"],
        "Q": spec["Q"],
        "T_alpha": spec.get("T_alpha", ""),
        "q_ratio": spec.get("q_ratio", ""),
        "alpha_l": spec.get("alpha_l", ""),
        "lambda": spec.get("lambda", ""),
        "certified_coverage": "",
        "avg_lower_bound": "",
        "runtime_total": "",
        "runtime_per_node": "",
        "num_nodes_certified": "",
        "checkpoint_path": "",
        "accuracy": "",
        "failure_reason": failure_reason,
    }
    if result:
        base.update(
            {
                "certified_coverage": result["certified_coverage"],
                "avg_lower_bound": result["avg_lower_bound"],
                "runtime_total": result["runtime_total"],
                "runtime_per_node": result["runtime_per_node"],
                "num_nodes_certified": result["num_nodes_certified"],
                "checkpoint_path": result["checkpoint_path"],
                "accuracy": result["accuracy"],
            }
        )
    return {field: base.get(field, "") for field in CSV_FIELDS[task]}


def write_config(output_dir, task, spec, result=None, failure_reason=""):
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    name_parts = [
        task,
        spec["dataset_key"],
        "l{}".format(spec["layers"]),
        "q{}".format(spec["q"]),
        "Q{}".format(spec["Q"]),
        "Ta{}".format(spec.get("T_alpha", 0)),
        "a{}".format(spec.get("alpha_l", "")),
        "lam{}".format(spec.get("lambda", "")),
    ]
    path = config_dir / "{}.json".format(slug("_".join(map(str, name_parts))))
    payload = {
        "spec": spec,
        "seed": int(args_global.seed),
        "device": str(DEVICE),
        "checkpoint": result["checkpoint_path"] if result else "",
        "train_info": result.get("train_info", {}) if result else {},
        "failure_reason": failure_reason,
        "note": (
            "The current robust_grn DGV code does not expose a separate T_alpha "
            "argument in certification; T_alpha is recorded for the requested "
            "ablation grid and can be wired into DGV in this script if added later."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def append_rows(csv_path, fieldnames, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_task(task, args, output_dir):
    specs = make_specs(task, args)
    if args.dry_run:
        for spec in specs:
            print(json.dumps(spec, sort_keys=True))
        return

    csv_path = output_dir / TASK_OUTPUTS[task]
    if csv_path.exists() and not args.append:
        csv_path.unlink()

    rows = []
    for spec in specs:
        print("Running {}: {}".format(task, spec))
        try:
            result = run_certification(spec, args, output_dir)
            write_config(output_dir, task, spec, result=result)
            rows.append(row_from_result(task, spec, result=result))
        except Exception as exc:
            failure = "{}: {}".format(exc.__class__.__name__, exc)
            print("FAILED {}: {}".format(spec, failure))
            write_config(output_dir, task, spec, failure_reason=failure)
            rows.append(row_from_result(task, spec, failure_reason=failure))
        append_rows(csv_path, CSV_FIELDS[task], rows[-1:])

    if task in ("q", "depth"):
        plot_task(task, csv_path, output_dir)


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def float_values(rows, key):
    values = []
    for row in rows:
        try:
            if row.get(key, "") != "":
                values.append(float(row[key]))
        except ValueError:
            pass
    return values


def plot_task(task, csv_path, output_dir):
    rows = [row for row in read_csv_rows(csv_path) if not row.get("failure_reason")]
    if not rows:
        return
    x_key = "q_ratio" if task == "q" else "layers"
    y_key = "certified_coverage"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_rows.sort(key=lambda row: float(row[x_key]))
        xs = [float(row[x_key]) for row in dataset_rows]
        ys = [float(row[y_key]) for row in dataset_rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=dataset)
    ax.set_xlabel("q ratio" if task == "q" else "GCNII layers")
    ax.set_ylabel("Certified coverage (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / ("q_sensitivity.png" if task == "q" else "depth_sensitivity.png"), dpi=220)
    plt.close(fig)


def summarize_direction(rows, x_key, y_key):
    rows = [row for row in rows if not row.get("failure_reason")]
    if len(rows) < 2:
        return "insufficient completed runs"
    grouped = {}
    for row in rows:
        try:
            x = float(row[x_key])
            y = float(row[y_key])
        except (TypeError, ValueError):
            continue
        grouped.setdefault(x, []).append(y)
    if len(grouped) < 2:
        return "insufficient completed runs"
    xs = sorted(grouped)
    first = np.mean(grouped[xs[0]])
    last = np.mean(grouped[xs[-1]])
    delta = last - first
    if abs(delta) < 1e-6:
        return "roughly unchanged ({:.2f} -> {:.2f})".format(first, last)
    direction = "increases" if delta > 0 else "decreases"
    return "{} on average ({:.2f} -> {:.2f})".format(direction, first, last)


def write_summary(output_dir):
    talpha_rows = read_csv_rows(output_dir / TASK_OUTPUTS["talpha"])
    q_rows = read_csv_rows(output_dir / TASK_OUTPUTS["q"])
    depth_rows = read_csv_rows(output_dir / TASK_OUTPUTS["depth"])
    hyper_rows = read_csv_rows(output_dir / TASK_OUTPUTS["gcnii_hyper"])

    completed = [row for row in talpha_rows + q_rows + depth_rows + hyper_rows if not row.get("failure_reason")]
    failures = [row for row in talpha_rows + q_rows + depth_rows + hyper_rows if row.get("failure_reason")]
    checkpoints = sorted({row.get("checkpoint_path", "") for row in completed if row.get("checkpoint_path")})
    retrained = []
    for config_path in sorted((output_dir / "configs").glob("*.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        train_info = config.get("train_info") or {}
        if train_info and not train_info.get("loaded_existing", True):
            retrained.append(
                {
                    "config": config_path.name,
                    "checkpoint": config.get("checkpoint", ""),
                    "train_info": train_info,
                }
            )

    summary = [
        "# Ablation Studies Summary",
        "",
        "- T_alpha: {}. Note: current robust_grn/DGV certification code has no exposed T_alpha control, so the script records the requested grid and keeps the certification path unchanged until that parameter is added to DGV.".format(
            summarize_direction(talpha_rows, "T_alpha", "certified_coverage")
        ),
        "- q sensitivity: certified coverage {} as q_ratio grows.".format(
            summarize_direction(q_rows, "q_ratio", "certified_coverage")
        ),
        "- Layer depth: certified coverage {} as depth grows; runtime per node {}.".format(
            summarize_direction(depth_rows, "layers", "certified_coverage"),
            summarize_direction(depth_rows, "layers", "runtime_per_node"),
        ),
        "- GCNII alpha_l/lambda: {} completed hyperparameter runs. Compare rows against alpha_l=0.1 and lambda=0.5 in gcnii_hyper_sensitivity.csv to support the default choice.".format(
            len([row for row in hyper_rows if not row.get("failure_reason")])
        ),
        "- Checkpoints used or generated: {}".format(", ".join(checkpoints) if checkpoints else "none recorded"),
        "- Retrained missing checkpoints: {}".format(len(retrained)),
        "- Failed runs: {}".format(len(failures)),
    ]
    if retrained:
        summary.append("")
        summary.append("## Retrained Checkpoints")
        for item in retrained[:20]:
            info = item["train_info"]
            summary.append(
                "- `{checkpoint}` from `{config}`: epochs_requested={epochs_requested}, patience={patience}, lr={lr}, wd1={wd1}, wd2={wd2}, dropout={dropout}, best_epoch={best_epoch}".format(
                    checkpoint=item["checkpoint"],
                    config=item["config"],
                    epochs_requested=info.get("epochs_requested", ""),
                    patience=info.get("patience", ""),
                    lr=info.get("lr", ""),
                    wd1=info.get("wd1", ""),
                    wd2=info.get("wd2", ""),
                    dropout=info.get("dropout", ""),
                    best_epoch=info.get("best_epoch", ""),
                )
            )
    if failures:
        summary.append("")
        summary.append("## Failures")
        for row in failures[:20]:
            summary.append(
                "- {dataset} L={layers} q={q} Q={Q}: {failure_reason}".format(**row)
            )
    path = output_dir / "ablation_summary.md"
    path.write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="result/ablation")
    parser.add_argument("--task", choices=["talpha", "q", "depth", "gcnii_hyper", "all"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append", action="store_true", help="append to existing CSV instead of replacing it")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=None, help="override default layers for talpha/q/gcnii_hyper")
    parser.add_argument("--Q-list", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--max-nodes", type=int, default=None, help="smoke-test limit for certified test nodes")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd1", type=float, default=0.01)
    parser.add_argument("--wd2", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--cert-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


args_global = None


def main():
    global args_global
    args = parse_args()
    args_global = args
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(ROBUST_GRN_DIR)

    tasks = ["talpha", "q", "depth", "gcnii_hyper"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, args, output_dir)
    if not args.dry_run:
        write_summary(output_dir)
        print("Saved ablation outputs under:", output_dir)


if __name__ == "__main__":
    main()
