import argparse
import csv
import json
import math
import os
import pickle as pkl
import random
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn
from torch.nn import Parameter
from tqdm import tqdm

from model import RobustGRNModel, Robustlinear
from process_nodes_classification import (
    load_npz,
    process_Amazon_CS,
    process_Amazon_Photo,
)
from utils import load_citation, normalized_adj_tensor, sparse_mx_to_torch_sparse_tensor, sys_normalized_adjacency


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


COMBOS = [
    {"dataset": "citeseer", "model": "gcn", "layers": 2, "q_max": 50, "checkpoint": None},
    {
        "dataset": "citeseer",
        "model": "gcnii",
        "layers": 32,
        "q_max": 50,
        "checkpoint": "pretrained/cd35a8a4fba44d6cb28ea899171d7497.pt",
    },
    {"dataset": "amazon_photo", "model": "gcnii", "layers": 8, "q_max": 100, "checkpoint": None},
    {"dataset": "amazon_cs", "model": "gcnii", "layers": 8, "q_max": 100, "checkpoint": None},
    {"dataset": "actor", "model": "gcnii", "layers": 8, "q_max": 100, "checkpoint": None},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adj_mm(adj, x):
    if getattr(adj, "is_sparse", False):
        return torch.spmm(adj, x)
    return adj.mm(x)


class RobustGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-bound, bound)

    def forward(self, input_tensor, adj):
        return adj_mm(adj, input_tensor).mm(self.weight)

    def bounds_continuous(self, input_tensor, input_lower, input_upper, adj):
        weight = self.weight
        w_plus = F.relu(weight)
        w_minus = F.relu(-weight)
        lower = adj.mm(input_lower.mm(w_plus) - input_upper.mm(w_minus))
        upper = adj.mm(input_upper.mm(w_plus) - input_lower.mm(w_minus))
        input_layer = adj.mm(input_tensor.mm(weight))
        return lower, upper, input_layer

    def slice_adj(self, adj, nodes):
        if len(nodes) == adj.shape[0]:
            return adj, nodes
        nonzero = adj[nodes].nonzero()
        nbs = torch.unique(nonzero[:, 1]).to(adj.device)
        adj_slice1 = adj[nodes]
        if len(nbs) == adj.shape[0]:
            return adj_slice1, nbs
        return adj_slice1.index_select(1, nbs), nbs

    def phi_backward(self, phi, adj, is_last=False):
        if is_last:
            phi_hat = torch.einsum("ij,ilm->iljm", adj, phi)
        else:
            phi_hat = torch.einsum("ijkl,km->ijml", phi, adj)
        return torch.tensordot(phi_hat, self.weight, dims=((3,), (1,)))

    def dual_backward(self, phi, nodes, bounds, adj, compute_objective=False, is_last=False):
        adj_slice, nbs = self.slice_adj(adj, nodes)
        lb, ub = bounds
        lb = lb[nbs]
        ub = ub[nbs]
        phi_hat = self.phi_backward(phi, adj_slice, is_last=is_last)
        phi_hat_plus = F.relu(phi_hat)
        phi_hat_minus = F.relu(-phi_hat)
        omega = ub / (ub - lb + 1e-9)
        crossing = ((lb < 0) & (ub > 0)).float()
        positive = ((lb > 0) & (ub > 0)).float()
        next_phi = phi_hat.mul(positive) + (phi_hat_plus.mul(omega) - phi_hat_minus.mul(omega)).mul(crossing)
        if compute_objective:
            objective = phi_hat_plus.mul(ub.mul(lb) / (ub - lb + 1e-9)).mul(crossing).sum((-2, -1))
        else:
            objective = None
        bias_objective = torch.zeros(phi.shape[0], phi.shape[1], device=phi.device)
        return next_phi, bias_objective, objective


class RobustGCNModel(nn.Module):
    def __init__(self, nfeat, adj, nlayers, hidden_dim, nclass, dropout):
        super().__init__()
        self.adj = adj
        self.nclass = nclass
        self.dropout = dropout
        self.fcs = nn.ModuleList([Robustlinear(nfeat, hidden_dim), Robustlinear(hidden_dim, nclass)])
        self.convs = nn.ModuleList([RobustGCNLayer(hidden_dim, hidden_dim) for _ in range(nlayers)])
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        hidden = F.relu(self.fcs[0](x))
        for conv in self.convs:
            hidden = F.dropout(hidden, self.dropout, training=self.training)
            hidden = F.relu(conv(hidden, adj))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return F.log_softmax(self.fcs[-1](hidden), dim=1)

    def predict(self, input_tensor, nodes):
        return self.forward(input_tensor, self.adj).max(-1)[1][nodes]

    def get_neighbors(self, nodes, adj):
        if len(nodes) == adj.shape[0]:
            return nodes
        nonzero = adj[nodes].nonzero()
        return torch.unique(nonzero[:, 1])

    def dual_backward(self, input_tensor, nodes, q, q_global, adj, return_perturbations=False):
        input_tensor = input_tensor.float()
        batch = len(nodes)
        bounds = []

        lower_bound, upper_bound, input_layer = self.fcs[0].bounds_binary(input_tensor, q, q_global)
        bounds.append((lower_bound, upper_bound))
        for layer in self.convs:
            lower_bound, upper_bound, input_layer = layer.bounds_continuous(
                input_layer,
                bounds[-1][0],
                bounds[-1][1],
                self.adj,
            )
            bounds.append((lower_bound, upper_bound))

        target_classes = self.predict(input_tensor, nodes)
        predicted_onehot = torch.eye(self.nclass, device=target_classes.device)[target_classes]
        c_tensor = predicted_onehot.unsqueeze(1) - torch.eye(self.nclass, device=predicted_onehot.device)
        phis = [-c_tensor]
        bias_terms = torch.zeros(batch, self.nclass, device=input_tensor.device)
        i_terms = torch.zeros(batch, self.nclass, device=input_tensor.device)

        phi = torch.einsum("ilm,mn->iln", phis[-1], self.fcs[-1].weight)
        phis.append(phi)

        nbs = nodes
        for layer_ix in np.arange(0, len(self.convs))[::-1]:
            layer = self.convs[layer_ix]
            phi = phis[-1]
            if layer_ix == len(self.convs) - 1:
                nbs = nodes
            else:
                nbs = self.get_neighbors(nbs, adj)
            ret = layer.dual_backward(
                phi,
                nbs,
                bounds[layer_ix],
                adj,
                compute_objective=True,
                is_last=(layer_ix == len(self.convs) - 1),
            )
            next_phi, bias_term, objective_term = ret
            phis.append(next_phi)
            bias_terms += bias_term
            if objective_term is not None:
                i_terms += objective_term

        phi_1_hat = self.fcs[0].phi_backward(phis[-1])
        bias_terms += self.fcs[0].bias_objective_term(phis[-1])
        nbs = self.get_neighbors(nbs, adj)
        input_hat = input_tensor[nbs]
        delta = F.relu(phi_1_hat).mul(1 - input_hat) + F.relu(-phi_1_hat).mul(input_hat)
        q_largest_local, _ = delta.topk(q, dim=3)
        q_largest_overall = q_largest_local.reshape([batch, self.nclass, -1])
        n_sel = min(q_global, q)
        q_largest, _ = q_largest_overall.topk(n_sel, -1)
        rho = q_largest[:, :, -1].unsqueeze(-1)
        eta = F.relu(q_largest_local[:, :, :, -1] - rho)
        psi_term = F.relu(delta - (rho + eta).unsqueeze(-1)).sum((2, 3))
        trace_term = input_hat.mul(phi_1_hat).sum((2, 3))
        final_objective = i_terms - bias_terms - trace_term - psi_term - q * eta.sum(-1) - q_global * rho.squeeze(-1)
        if return_perturbations:
            return final_objective, None
        return final_objective


def load_citation_degree(dataset):
    graph_path = Path("dataset") / dataset.title() / "raw" / "ind.{}.graph".format(dataset)
    with graph_path.open("rb") as f:
        graph = pkl.load(f, encoding="latin1")
    g = nx.from_dict_of_lists(graph)
    return np.array([g.degree(i) for i in range(len(graph))], dtype=np.int64)


def load_npz_degree(path):
    dataset = load_npz(path)
    adj = dataset["adj_matrix"] if "adj_matrix" in dataset else sp.csr_matrix(
        (dataset["adj_data"], dataset["adj_indices"], dataset["adj_indptr"])
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)


def load_actor_degree():
    from torch_geometric.datasets import Actor

    data = Actor(root="./dataset")[0]
    edge_index = data.edge_index.cpu().numpy()
    n = int(data.num_nodes)
    adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)


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
    adj_sparse = sp.csr_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
    adj_sparse = adj_sparse + adj_sparse.T.multiply(adj_sparse.T > adj_sparse) - adj_sparse.multiply(adj_sparse.T > adj_sparse)
    adj = sys_normalized_adjacency(adj_sparse)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, idx_train, idx_val, idx_test, dense_adj


def load_data(dataset):
    if dataset == "citeseer":
        adj, features, labels, idx_train, idx_val, idx_test, dense_adj = load_citation("citeseer")
        degrees = load_citation_degree("citeseer")
    elif dataset == "amazon_photo":
        adj, features, labels, idx_train, idx_val, idx_test, dense_adj = process_Amazon_Photo()
        degrees = load_npz_degree("dataset/other/amazon_electronics_photo.npz")
    elif dataset == "amazon_cs":
        adj, features, labels, idx_train, idx_val, idx_test, dense_adj = process_Amazon_CS()
        degrees = load_npz_degree("dataset/other/amazon_cs.npz")
    elif dataset == "actor":
        adj, features, labels, idx_train, idx_val, idx_test, dense_adj = process_actor_local()
        degrees = load_actor_degree()
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset))
    return adj, features, labels, idx_train, idx_val, idx_test, dense_adj, degrees


def split_names(num_nodes, idx_train, idx_val, idx_test):
    names = np.array(["other"] * num_nodes, dtype=object)
    names[np.asarray(idx_train.cpu() if torch.is_tensor(idx_train) else idx_train, dtype=np.int64)] = "train"
    names[np.asarray(idx_val.cpu() if torch.is_tensor(idx_val) else idx_val, dtype=np.int64)] = "val"
    names[np.asarray(idx_test.cpu() if torch.is_tensor(idx_test) else idx_test, dtype=np.int64)] = "test"
    return names


def build_model(combo, features, labels, dense_adj, args):
    nclass = int(labels.max()) + 1
    if combo["model"] == "gcn":
        return RobustGCNModel(
            nfeat=features.shape[1],
            adj=dense_adj,
            nlayers=combo["layers"],
            hidden_dim=args.hidden_dim,
            nclass=nclass,
            dropout=args.dropout,
        )
    return RobustGRNModel(
        nfeat=features.shape[1],
        adj=dense_adj,
        nlayers=combo["layers"],
        dim=[args.hidden_dim],
        nclass=nclass,
        dropout=args.dropout,
        lamda=args.lamda,
        alpha=args.alpha,
        variant=False,
    )


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return preds.eq(labels).double().sum() / len(labels)


def train_model(model, features, adj, labels, idx_train, idx_val, checkpoint_path, args):
    optimizer = torch.optim.Adam(
        [
            {"params": model.params1, "weight_decay": args.wd1},
            {"params": model.params2, "weight_decay": args.wd2},
        ],
        lr=args.lr,
    )
    best = float("inf")
    bad_counter = 0
    best_epoch = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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
            acc_val = accuracy(output[idx_val], labels[idx_val])
        if loss_val.item() < best:
            best = loss_val.item()
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_counter += 1
        if (epoch + 1) % args.log_every == 0:
            print(
                "epoch={} train_loss={:.4f} val_loss={:.4f} val_acc={:.4f}".format(
                    epoch + 1,
                    loss_train.item(),
                    loss_val.item(),
                    acc_val.item(),
                )
            )
        if bad_counter == args.patience:
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return {"best_epoch": best_epoch, "best_val_loss": best}


def load_or_train(combo, model, features, adj, labels, idx_train, idx_val, out_dir, args):
    candidate = combo.get("checkpoint")
    checkpoint_path = Path(candidate) if candidate else out_dir / "checkpoints" / "{}_{}_l{}.pt".format(
        combo["dataset"],
        combo["model"],
        combo["layers"],
    )
    if checkpoint_path.exists() and not args.retrain:
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print("Loaded checkpoint:", checkpoint_path)
            return str(checkpoint_path), {"loaded_existing": True}
        except Exception as exc:
            print("Could not load checkpoint {}; retraining. Error: {}".format(checkpoint_path, exc))
    info = train_model(model, features, adj, labels, idx_train, idx_val, checkpoint_path, args)
    info["loaded_existing"] = False
    return str(checkpoint_path), info


def certify_robust_for_q(model, features, nodes, q_local, q_global, dense_adj, batch_size):
    robust = np.zeros(features.shape[0], dtype=bool)
    model.eval()
    node_chunks = [nodes[i : i + batch_size] for i in range(0, len(nodes), batch_size)]
    for chunk in tqdm(node_chunks, desc="Q={}".format(q_global), leave=False):
        chunk_device = chunk.to(features.device)
        with torch.no_grad():
            lb = model.dual_backward(features, chunk_device, q_local, q_global, dense_adj)
        robust_chunk = ((lb.detach().cpu().numpy() > 0).sum(1) == model.nclass - 1)
        robust[chunk.cpu().numpy()] = robust_chunk
    return robust


def compute_maxq(model, features, q_local, q_values, dense_adj, batch_size, max_nodes=None):
    n = features.shape[0]
    nodes = torch.arange(n)
    if max_nodes is not None:
        nodes = nodes[:max_nodes]
    max_q = np.zeros(n, dtype=np.int64)
    for q_global in q_values:
        robust = certify_robust_for_q(model, features, nodes, q_local, q_global, dense_adj, batch_size)
        selected = nodes.cpu().numpy()
        max_q[selected[robust[selected]]] = q_global
    return max_q


def write_node_csv(path, dataset, model_name, degrees, max_q, split, correct, pred, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "node_id", "degree", "max_q", "split", "correct", "clean_pred", "label"],
        )
        writer.writeheader()
        for node_id in range(len(max_q)):
            writer.writerow(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "node_id": node_id,
                    "degree": int(degrees[node_id]),
                    "max_q": int(max_q[node_id]),
                    "split": split[node_id],
                    "correct": int(correct[node_id]),
                    "clean_pred": int(pred[node_id]),
                    "label": int(labels[node_id]),
                }
            )


def read_node_csv(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def mean_by_degree(rows):
    grouped = {}
    for row in rows:
        deg = int(row["degree"])
        grouped.setdefault(deg, []).append(float(row["max_q"]))
    xs = np.array(sorted(grouped), dtype=float)
    ys = np.array([np.mean(grouped[int(x)]) for x in xs], dtype=float)
    return xs, ys


def plot_panel(rows, title, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    degrees = np.array([int(r["degree"]) for r in rows], dtype=float)
    max_q = np.array([int(r["max_q"]) for r in rows], dtype=float)
    positive = degrees > 0
    degrees = degrees[positive]
    max_q = max_q[positive]
    rows_positive = [r for r, keep in zip(rows, positive) if keep]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(degrees, max_q, s=14, alpha=0.25, color="#1f77b4", edgecolors="none")
    mean_x, mean_y = mean_by_degree(rows_positive)
    mean_positive = mean_x > 0
    ax.plot(mean_x[mean_positive], mean_y[mean_positive], color="orange", linewidth=2.5, label="mean Max-Q")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Degree", fontsize=13)
    ax.set_ylabel("Max-Q robust", fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined(panel_specs, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()
    for ax, spec in zip(axes, panel_specs):
        rows = read_node_csv(spec["csv"])
        degrees = np.array([int(r["degree"]) for r in rows], dtype=float)
        max_q = np.array([int(r["max_q"]) for r in rows], dtype=float)
        positive = degrees > 0
        rows_positive = [r for r, keep in zip(rows, positive) if keep]
        ax.scatter(degrees[positive], max_q[positive], s=10, alpha=0.22, color="#1f77b4", edgecolors="none")
        mean_x, mean_y = mean_by_degree(rows_positive)
        mean_positive = mean_x > 0
        ax.plot(mean_x[mean_positive], mean_y[mean_positive], color="orange", linewidth=2.2, label="mean Max-Q")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Max-Q robust")
        ax.set_title(spec["title"])
        ax.legend(fontsize=10)
    for ax in axes[len(panel_specs) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def summarize(csv_paths, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "num_nodes", "spearman_rho", "p_value"])
        writer.writeheader()
        for path in csv_paths:
            rows = read_node_csv(path)
            degree = np.array([int(r["degree"]) for r in rows], dtype=float)
            max_q = np.array([int(r["max_q"]) for r in rows], dtype=float)
            if len(np.unique(degree)) <= 1 or len(np.unique(max_q)) <= 1:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(degree, max_q)
            writer.writerow(
                {
                    "dataset": rows[0]["dataset"],
                    "model": rows[0]["model"],
                    "num_nodes": len(rows),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )


def run_combo(combo, args, out_dir):
    print("Running combo:", combo)
    set_seed(args.seed)
    adj, features, labels, idx_train, idx_val, idx_test, dense_adj, degrees = load_data(combo["dataset"])
    features = features.float().to(DEVICE)
    labels = labels.long().to(DEVICE)
    adj = adj.to(DEVICE)
    dense_adj = dense_adj.float().to(DEVICE)
    idx_train = idx_train.long().to(DEVICE)
    idx_val = idx_val.long().to(DEVICE)
    idx_test = idx_test.long().to(DEVICE)

    model = build_model(combo, features, labels, dense_adj, args).to(DEVICE)
    checkpoint, train_info = load_or_train(combo, model, features, adj, labels, idx_train, idx_val, out_dir, args)
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        pred = output.argmax(dim=1)
    correct = pred.cpu().eq(labels.cpu()).numpy()
    split = split_names(features.shape[0], idx_train.cpu(), idx_val.cpu(), idx_test.cpu())

    q_local = max(1, int(math.floor(0.01 * features.shape[1])))
    q_max = min(combo["q_max"], args.q_limit) if args.q_limit is not None else combo["q_max"]
    q_values = list(range(1, q_max + 1))
    start = time.time()
    max_q = compute_maxq(
        model,
        features,
        q_local,
        q_values,
        dense_adj,
        args.cert_batch_size,
        max_nodes=args.max_nodes,
    )
    runtime = time.time() - start

    model_label = combo["model"].upper() if combo["model"] == "gcn" else "GCNII"
    csv_path = out_dir / "csv" / "{}_{}_degree_maxq.csv".format(combo["dataset"], combo["model"])
    write_node_csv(csv_path, combo["dataset"], model_label, degrees, max_q, split, correct, pred.cpu().numpy(), labels.cpu().numpy())
    config = {
        "combo": combo,
        "checkpoint": checkpoint,
        "train_info": train_info,
        "q_local": q_local,
        "q_values": q_values,
        "runtime_seconds": runtime,
        "args": vars(args),
        "device": str(DEVICE),
    }
    config_path = out_dir / "configs" / "{}_{}_config.json".format(combo["dataset"], combo["model"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    panel_title = "{} + {}".format(combo["dataset"], model_label)
    plot_panel(read_node_csv(csv_path), panel_title, out_dir / "figures" / "{}_{}_degree_maxq.png".format(combo["dataset"], combo["model"]))
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="result/figure8_degree_maxq")
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
    parser.add_argument("--cert-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Subset like citeseer:gcn amazon_photo:gcnii")
    parser.add_argument("--max-nodes", type=int, default=None, help="Smoke-test limit; omit for full Figure 8")
    parser.add_argument("--q-limit", type=int, default=None, help="Smoke-test Q cap; omit for requested full Q range")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.only:
        wanted = set(args.only)
        combos = [c for c in COMBOS if "{}:{}".format(c["dataset"], c["model"]) in wanted]
    else:
        combos = COMBOS
    csv_paths = [out_dir / "csv" / "{}_{}_degree_maxq.csv".format(c["dataset"], c["model"]) for c in combos]
    if not args.plot_only:
        csv_paths = []
        for combo in combos:
            csv_paths.append(run_combo(combo, args, out_dir))
    summarize(csv_paths, out_dir / "degree_maxq_spearman_summary.csv")
    panel_specs = []
    for combo, csv_path in zip(combos, csv_paths):
        label = combo["model"].upper() if combo["model"] == "gcn" else "GCNII"
        panel_specs.append({"csv": csv_path, "title": "{} + {}".format(combo["dataset"], label)})
    plot_combined(panel_specs, out_dir / "figure8_degree_maxq_5panel.png")
    print("Saved outputs under:", out_dir)


if __name__ == "__main__":
    main()
