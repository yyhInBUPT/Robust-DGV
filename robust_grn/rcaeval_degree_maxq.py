import argparse
import csv
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.sparse import load_npz
from tqdm import tqdm

from model import RobustGRNModel
from utils import load_rcaeval_ob, load_rcaeval_ss, load_rcaeval_tt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "rcaeval_ob": {
        "system": "Online Boutique",
        "loader": load_rcaeval_ob,
        "dataset_dir": "RCAEval_OB",
        "checkpoint": "pretrained/rcaeval_ob_gcnii_l4.pt",
        "csv_name": "rcaeval_ob_degree_maxq.csv",
        "figure_name": "rcaeval_ob_degree_maxq.png",
    },
    "rcaeval_ss": {
        "system": "Sock Shop",
        "loader": load_rcaeval_ss,
        "dataset_dir": "RCAEval_SS",
        "checkpoint": "pretrained/rcaeval_ss_gcnii_l4.pt",
        "csv_name": "rcaeval_ss_degree_maxq.csv",
        "figure_name": "rcaeval_ss_degree_maxq.png",
    },
    "rcaeval_tt": {
        "system": "Train Ticket",
        "loader": load_rcaeval_tt,
        "dataset_dir": "RCAEval_TT",
        "checkpoint": "pretrained/rcaeval_tt_gcnii_l4.pt",
        "csv_name": "rcaeval_tt_degree_maxq.csv",
        "figure_name": "rcaeval_tt_degree_maxq.png",
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_names(num_nodes, idx_train, idx_val, idx_test):
    names = np.array(["other"] * num_nodes, dtype=object)
    names[idx_train.cpu().numpy()] = "train"
    names[idx_val.cpu().numpy()] = "val"
    names[idx_test.cpu().numpy()] = "test"
    return names


def load_node_meta(dataset_dir):
    path = SCRIPT_DIR / "dataset" / dataset_dir / "node_meta.json"
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    meta = {}
    for item in raw:
        node_id = item.get("global_node_id", item.get("node_id"))
        meta[int(node_id)] = {
            "case_id": int(item["case_id"]),
            "service_name": item["service"],
        }
    return meta


def load_degrees_without_self_loops(dataset_dir):
    adj_path = SCRIPT_DIR / "dataset" / dataset_dir / "adj.npz"
    adj = load_npz(adj_path).astype(np.float32).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)


def build_model(features, labels, model_adj, args):
    return RobustGRNModel(
        nfeat=features.shape[1],
        adj=model_adj,
        nlayers=4,
        dim=[args.hidden_dim],
        nclass=int(labels.max()) + 1,
        dropout=args.dropout,
        lamda=args.lamda,
        alpha=args.alpha,
        variant=False,
    ).to(DEVICE)


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
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        train_labels = labels[idx_train]
        class_count = torch.bincount(train_labels, minlength=int(labels.max()) + 1).float()
        class_weight = class_count.sum() / (len(class_count) * class_count)
        class_weight = class_weight.to(DEVICE)
        loss = F.nll_loss(output[idx_train], labels[idx_train], weight=class_weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            val_loss = F.nll_loss(output[idx_val], labels[idx_val])
        if val_loss.item() < best:
            best = val_loss.item()
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_counter += 1
        if (epoch + 1) % args.log_every == 0:
            print("epoch={} train_loss={:.4f} val_loss={:.4f}".format(epoch + 1, loss.item(), val_loss.item()))
        if bad_counter == args.patience:
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return {"loaded_existing": False, "best_epoch": best_epoch, "best_val_loss": best}


def load_or_train_model(dataset_key, spec, model, features, adj, labels, idx_train, idx_val, args):
    checkpoint_path = Path(spec["checkpoint"])
    if checkpoint_path.exists() and not args.retrain:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        return str(checkpoint_path), {"loaded_existing": True}
    checkpoint_path = Path(args.output_dir) / "checkpoints" / "{}_gcnii_l4.pt".format(dataset_key)
    info = train_model(model, features, adj, labels, idx_train, idx_val, checkpoint_path, args)
    return str(checkpoint_path), info


def certify_robust(model, features, nodes, q_local, q_global, certify_adj, batch_size):
    robust = np.zeros(features.shape[0], dtype=bool)
    chunks = [nodes[i : i + batch_size] for i in range(0, len(nodes), batch_size)]
    model.eval()
    for chunk in tqdm(chunks, desc="Q={}".format(q_global), leave=False):
        chunk_device = chunk.to(features.device)
        with torch.no_grad():
            lower_bounds = model.dual_backward(features, chunk_device, q_local, q_global, certify_adj)
        robust_chunk = ((lower_bounds.detach().cpu().numpy() > 0).sum(1) == model.nclass - 1)
        robust[chunk.cpu().numpy()] = robust_chunk
    return robust


def compute_maxq(model, features, nodes, q_local, q_values, certify_adj, batch_size, max_nodes=None):
    if max_nodes is not None:
        nodes = nodes[:max_nodes]
    max_q = np.zeros(features.shape[0], dtype=np.int64)
    selected = nodes.cpu().numpy()
    for q_global in q_values:
        robust = certify_robust(model, features, nodes, q_local, q_global, certify_adj, batch_size)
        max_q[selected[robust[selected]]] = q_global
    return max_q, nodes


def write_node_csv(path, dataset_key, system, nodes, degrees, meta, split, labels, pred, correct, max_q):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "system",
                "node_id",
                "case_id",
                "service_name",
                "split",
                "degree",
                "label",
                "clean_pred",
                "correct",
                "root_or_nonroot",
                "max_q",
            ],
        )
        writer.writeheader()
        for node_id in nodes.cpu().numpy():
            label = int(labels[node_id])
            writer.writerow({
                "dataset": dataset_key,
                "system": system,
                "node_id": int(node_id),
                "case_id": meta[int(node_id)]["case_id"],
                "service_name": meta[int(node_id)]["service_name"],
                "split": split[int(node_id)],
                "degree": int(degrees[int(node_id)]),
                "label": label,
                "clean_pred": int(pred[int(node_id)]),
                "correct": int(correct[int(node_id)]),
                "root_or_nonroot": "root" if label == 1 else "non-root",
                "max_q": int(max_q[int(node_id)]),
            })


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    if value is None:
        return "NA"
    try:
        if np.isnan(value):
            return "NA"
    except TypeError:
        pass
    return float(value)


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return None, None, None
    mean = float(values.mean())
    if n == 1:
        return mean, mean, mean
    sd = float(values.std(ddof=1))
    half_width = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return mean, mean - half_width, mean + half_width


def choose_degree_groups(degree):
    n = len(degree)
    unique = np.unique(degree)
    if n < 3:
        return np.array(["low"] * n, dtype=object), "insufficient_n"
    if len(unique) < 3:
        labels = np.array(["low"] * n, dtype=object)
        if len(unique) == 2:
            labels[degree == unique[1]] = "high"
            return labels, "unique_degree_groups_lt3"
        return labels, "constant_degree"
    order = np.argsort(degree, kind="mergesort")
    labels = np.empty(n, dtype=object)
    first = n // 3
    second = 2 * n // 3
    labels[order[:first]] = "low"
    labels[order[first:second]] = "medium"
    labels[order[second:]] = "high"
    return labels, "degree_rank_tertiles"


def safe_spearman(degree, max_q):
    if len(degree) < 3 or len(np.unique(degree)) < 2 or len(np.unique(max_q)) < 2:
        return None, None
    rho, p_value = stats.spearmanr(degree, max_q)
    if np.isnan(rho):
        return None, None
    return float(rho), float(p_value)


def safe_group_tests(groups, max_q):
    values = [max_q[groups == name] for name in ("low", "medium", "high") if np.any(groups == name)]
    if len(values) < 2:
        return None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            anova_f, anova_p = stats.f_oneway(*values)
        if np.isnan(anova_p):
            anova_p = None
    except Exception:
        anova_p = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, kruskal_p = stats.kruskal(*values)
        if np.isnan(kruskal_p):
            kruskal_p = None
    except Exception:
        kruskal_p = None
    return anova_p, kruskal_p


def trend_summary(group_stats):
    valid = [(name, group_stats[name][0]) for name in ("low", "medium", "high") if group_stats[name][0] is not None]
    if len(valid) < 2:
        return "insufficient groups"
    means = [value for _, value in valid]
    if max(means) - min(means) < 1e-9:
        return "weak/no clear trend"
    order = sorted(valid, key=lambda item: item[1], reverse=True)
    short = {"low": "L", "medium": "M", "high": "H"}
    return " > ".join(short[name] for name, _ in order)


def summarize_one_scope(rows, eval_scope, group_scope):
    filtered = [row for row in rows if row["split"] == "test"]
    if eval_scope == "correct_test_nodes":
        filtered = [row for row in filtered if int(row["correct"]) == 1]
    if group_scope == "root":
        filtered = [row for row in filtered if row["root_or_nonroot"] == "root"]
    elif group_scope == "non-root":
        filtered = [row for row in filtered if row["root_or_nonroot"] == "non-root"]

    base = {
        "dataset": rows[0]["dataset"],
        "system": rows[0]["system"],
        "eval_scope": eval_scope,
        "group_scope": group_scope,
        "n_nodes": len(filtered),
        "spearman_rho": "NA",
        "spearman_p": "NA",
        "anova_p": "NA",
        "kruskal_p": "NA",
        "grouping_method": "insufficient_n",
        "low_mean": "NA",
        "low_ci95_low": "NA",
        "low_ci95_high": "NA",
        "medium_mean": "NA",
        "medium_ci95_low": "NA",
        "medium_ci95_high": "NA",
        "high_mean": "NA",
        "high_ci95_low": "NA",
        "high_ci95_high": "NA",
        "trend_summary": "insufficient data",
    }
    if len(filtered) == 0:
        return base

    degree = np.array([int(row["degree"]) for row in filtered], dtype=float)
    max_q = np.array([int(row["max_q"]) for row in filtered], dtype=float)
    rho, spearman_p = safe_spearman(degree, max_q)
    groups, grouping_method = choose_degree_groups(degree)
    anova_p, kruskal_p = safe_group_tests(groups, max_q)
    group_stats = {}
    for name in ("low", "medium", "high"):
        mean, ci_low, ci_high = mean_ci95(max_q[groups == name])
        group_stats[name] = (mean, ci_low, ci_high)

    base.update({
        "spearman_rho": safe_float(rho),
        "spearman_p": safe_float(spearman_p),
        "anova_p": safe_float(anova_p),
        "kruskal_p": safe_float(kruskal_p),
        "grouping_method": grouping_method,
        "low_mean": safe_float(group_stats["low"][0]),
        "low_ci95_low": safe_float(group_stats["low"][1]),
        "low_ci95_high": safe_float(group_stats["low"][2]),
        "medium_mean": safe_float(group_stats["medium"][0]),
        "medium_ci95_low": safe_float(group_stats["medium"][1]),
        "medium_ci95_high": safe_float(group_stats["medium"][2]),
        "high_mean": safe_float(group_stats["high"][0]),
        "high_ci95_low": safe_float(group_stats["high"][1]),
        "high_ci95_high": safe_float(group_stats["high"][2]),
        "trend_summary": trend_summary(group_stats),
    })
    return base


def write_summary(out_dir, dataset_keys):
    rows_out = []
    for key in dataset_keys:
        csv_path = out_dir / "csv" / DATASETS[key]["csv_name"]
        if not csv_path.exists():
            continue
        rows = read_rows(csv_path)
        for eval_scope in ("all_test_nodes", "correct_test_nodes"):
            for group_scope in ("all", "root", "non-root"):
                rows_out.append(summarize_one_scope(rows, eval_scope, group_scope))
    if not rows_out:
        return
    path = out_dir / "rcaeval_degree_maxq_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def mean_by_degree(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["degree"]), []).append(float(row["max_q"]))
    xs = np.array(sorted(grouped), dtype=float)
    ys = np.array([np.mean(grouped[int(x)]) for x in xs], dtype=float)
    return xs, ys


def plot_dataset(rows, title, output_path, eval_scope):
    filtered = [row for row in rows if row["split"] == "test"]
    if eval_scope == "correct_test_nodes":
        filtered = [row for row in filtered if int(row["correct"]) == 1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if filtered:
        degree = np.array([int(row["degree"]) for row in filtered], dtype=float)
        max_q = np.array([int(row["max_q"]) for row in filtered], dtype=float)
        ax.scatter(degree, max_q, s=22, alpha=0.35, color="#1f77b4", edgecolors="none")
        mean_x, mean_y = mean_by_degree(filtered)
        ax.plot(mean_x, mean_y, color="orange", linewidth=2.5, label="mean Max-Q")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Max-Q robust")
    ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_all(out_dir, dataset_keys):
    panel_items = []
    for key in dataset_keys:
        csv_path = out_dir / "csv" / DATASETS[key]["csv_name"]
        if not csv_path.exists():
            continue
        rows = read_rows(csv_path)
        plot_dataset(
            rows,
            "{} ({})".format(DATASETS[key]["system"], "correct test"),
            out_dir / "figures" / DATASETS[key]["figure_name"],
            "correct_test_nodes",
        )
        plot_dataset(
            rows,
            "{} ({})".format(DATASETS[key]["system"], "all test"),
            out_dir / "figures" / DATASETS[key]["figure_name"].replace(".png", "_all_test_nodes.png"),
            "all_test_nodes",
        )
        plot_dataset(
            rows,
            "{} ({})".format(DATASETS[key]["system"], "correct test"),
            out_dir / "figures" / DATASETS[key]["figure_name"].replace(".png", "_correct_test_nodes.png"),
            "correct_test_nodes",
        )
        panel_items.append((key, rows))

    if not panel_items:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (key, rows) in zip(axes, panel_items):
        filtered = [row for row in rows if row["split"] == "test" and int(row["correct"]) == 1]
        if filtered:
            degree = np.array([int(row["degree"]) for row in filtered], dtype=float)
            max_q = np.array([int(row["max_q"]) for row in filtered], dtype=float)
            ax.scatter(degree, max_q, s=18, alpha=0.35, color="#1f77b4", edgecolors="none")
            mean_x, mean_y = mean_by_degree(filtered)
            ax.plot(mean_x, mean_y, color="orange", linewidth=2.2, label="mean Max-Q")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Max-Q robust")
        ax.set_title(DATASETS[key]["system"])
        ax.legend(fontsize=9)
    for ax in axes[len(panel_items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_rcaeval_degree_maxq_3panel.png", dpi=250)
    plt.close(fig)


def run_dataset(key, spec, args, out_dir):
    print("Running", key)
    set_seed(args.seed)
    adj, features, labels, idx_train, idx_val, idx_test, certify_adj_sparse = spec["loader"]()
    degrees = load_degrees_without_self_loops(spec["dataset_dir"])
    meta = load_node_meta(spec["dataset_dir"])
    split = split_names(features.shape[0], idx_train, idx_val, idx_test)

    features = features.float().to(DEVICE)
    labels = labels.long().to(DEVICE)
    adj = adj.to(DEVICE)
    certify_adj = certify_adj_sparse.to_dense().float().to(DEVICE) if certify_adj_sparse.is_sparse else certify_adj_sparse.float().to(DEVICE)
    idx_train = idx_train.long().to(DEVICE)
    idx_val = idx_val.long().to(DEVICE)
    idx_test = idx_test.long().to(DEVICE)

    model = build_model(features, labels, adj, args)
    checkpoint, train_info = load_or_train_model(key, spec, model, features, adj, labels, idx_train, idx_val, args)
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        pred = output.argmax(dim=1)
    correct = pred.cpu().eq(labels.cpu()).numpy()

    q_values = list(range(1, args.q_max + 1))
    start = time.time()
    max_q, evaluated_nodes = compute_maxq(
        model,
        features,
        idx_test.cpu(),
        args.q,
        q_values,
        certify_adj,
        args.cert_batch_size,
        max_nodes=args.max_nodes,
    )
    runtime = time.time() - start

    csv_path = out_dir / "csv" / spec["csv_name"]
    write_node_csv(
        csv_path,
        key,
        spec["system"],
        evaluated_nodes,
        degrees,
        meta,
        split,
        labels.cpu().numpy(),
        pred.cpu().numpy(),
        correct,
        max_q,
    )
    return {
        "dataset": key,
        "system": spec["system"],
        "checkpoint": checkpoint,
        "train_info": train_info,
        "num_evaluated_nodes": int(len(evaluated_nodes)),
        "runtime_seconds": runtime,
        "csv": str(csv_path),
    }


def write_config(out_dir, args, run_infos, dataset_keys):
    config = {
        "dataset_list": dataset_keys,
        "model_architecture": "GCNII / RobustGRNModel",
        "layers": 4,
        "training_hyperparameters": {
            "seed": args.seed,
            "training_mode": "full-batch transductive",
            "epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "weight_decay_conv": args.wd1,
            "weight_decay_fc": args.wd2,
            "alpha_l": args.alpha,
            "beta_l": "log(lambda / l + 1)",
            "lambda": args.lamda,
        },
        "checkpoint_paths": {info["dataset"]: info["checkpoint"] for info in run_infos},
        "q": args.q,
        "Q_scan_range": list(range(1, args.q_max + 1)),
        "evaluated_split": "test",
        "eval_scopes": ["all_test_nodes", "correct_test_nodes"],
        "random_seed": args.seed,
        "code_command_used": " ".join(sys.argv),
        "run_infos": run_infos,
    }
    path = out_dir / "configs" / "rcaeval_degree_maxq_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="result/rcaeval_degree_maxq")
    parser.add_argument("--only", nargs="*", choices=sorted(DATASETS), default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd1", type=float, default=0.01)
    parser.add_argument("--wd2", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--q-max", type=int, default=5)
    parser.add_argument("--cert-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-nodes", type=int, default=None, help="Smoke-test limit for test nodes")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_keys = args.only if args.only else list(DATASETS)
    run_infos = []
    if not args.plot_only:
        for key in dataset_keys:
            run_infos.append(run_dataset(key, DATASETS[key], args, out_dir))
        write_config(out_dir, args, run_infos, dataset_keys)
    write_summary(out_dir, dataset_keys)
    plot_all(out_dir, dataset_keys)
    print("Saved RCAEval degree-MaxQ outputs under:", out_dir)


if __name__ == "__main__":
    main()
