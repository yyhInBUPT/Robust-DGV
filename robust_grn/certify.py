import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm,trange

from robust_grn.model import RobustGRNModel
from robust_grn.utils import load_citation, load_rcaeval_ob, load_rcaeval_ss, load_rcaeval_tt


def chunker(seq, size):##块处理数据
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def certify(model, attrs,q, A, nodes=None, Q=12, optimize_steps=5,batch_size = 8,
           certify_nonrobustness=False, progress=True):

    N = attrs.shape[0]
    K = model.nclass


    if nodes is None:
        nodes = torch.arange(N)
    else:
        nodes = torch.as_tensor(nodes).long().cpu()

    lower_bounds = []
    nonrobust_nodes = np.zeros(N, dtype=bool)
    # batch = np.array([0,1,2,3,4,5,6,7])

    _iter = chunker(nodes, batch_size)
    if progress:
        _iter = tqdm(_iter, total=int(np.ceil(len(nodes) / batch_size)))


    for chunk in _iter:
        chunk_device = chunk.to(attrs.device)
        lb, pert = model.dual_backward(attrs, chunk_device, q, Q, A,return_perturbations=True)# ,pert
        lb = lb.detach()
        lower_bounds.append(lb.cpu().numpy())
        if certify_nonrobustness:
            predicted_before = model.predict(attrs, chunk_device)
            predicted_after = []
            for ix, node in enumerate(chunk_device):
                attack_successful = []
                for cl in torch.sort(lb[ix])[1]:
                    if lb[ix, cl] >= 0:
                        # only test for nonrobustness when we cannot certify robustness
                        attack_successful.append(False)
                        continue

                    pert_ixs = tuple(torch.tensor(pert[ix, cl].T, device=attrs.device).long())
                    attrs.index_put_(pert_ixs, 1 - attrs[pert_ixs])
                    # nd = node.item()
                    # a = model.predict(attrs, [node]).cpu().numpy()

                    after = model.predict(attrs, torch.tensor([int(node.item())], device=attrs.device)).cpu().numpy()

                    success = bool(np.any(after != predicted_before[ix].cpu().numpy()))
                    attack_successful.append(success)
                    attrs.index_put_(pert_ixs, 1 - attrs[pert_ixs])

                    if success:
                        # once we have an adversarial example for a single class we can stop
                        break

                predicted_after.append(np.any(attack_successful))
            predicted_after = np.row_stack(predicted_after)
            nonrobust_nodes[chunk.numpy()] = predicted_after[:, 0].astype(bool)
        # lower_bounds = np.row_stack(lower_bounds)
        # # print(lower_bounds[0])
        # robust_nodes = ((lower_bounds > 0).sum(1) == K - 1)



    lower_bounds = np.row_stack(lower_bounds)
    # print(lower_bounds[0])
    robust_nodes = np.zeros(N, dtype=bool)
    robust_nodes[nodes.numpy()] = ((lower_bounds > 0).sum(1) == K - 1)

    return robust_nodes, nonrobust_nodes
def certify_Max_Q(model, attrs,q,Max_Q, A, batch_size = 1
           , progress=True):
    N = attrs.shape[0]
    K = model.nclass
    Q = np.arange(100)
    nonrobust_nodes = np.zeros(N)
    lower_bounds=[]
    _iter = chunker(torch.arange(N), batch_size)
    if progress:
        _iter = tqdm(_iter, total=int(np.ceil(N / batch_size)))
    for chunk in _iter:
        predicted_before = model.predict(attrs, chunk)
        for Q_ in Q:
            lb, pert = model.dual_backward(attrs, chunk, q, Q_, A, return_perturbations=True)
            lb = lb.detach()
            lower_bounds.append(lb.cpu().numpy())
            for ix, node in enumerate(chunk):
                attack_successful = []
                for cl in torch.sort(lb[ix])[1]:
                    if lb[ix, cl] >= 0:
                        # only test for nonrobustness when we cannot certify robustness
                        continue

                    pert_ixs = tuple(torch.tensor(pert[ix, cl].T))
                    attrs.index_put_(pert_ixs, 1 - attrs[pert_ixs])


                    after = model.predict(attrs, [node]).cpu().numpy()

                    success = after != predicted_before[ix].cpu().numpy()
                    attack_successful.append(success)
                    attrs.index_put_(pert_ixs, 1 - attrs[pert_ixs])

                    if success:
                        # once we have an adversarial example for a single class we can stop
                        break

                predicted_after.append(np.any(attack_successful))
            predicted_after = np.row_stack(predicted_after)
            nonrobust_nodes[chunk] = predicted_after[:, 0]
    lower_bounds = np.row_stack(lower_bounds)

    robust_nodes = ((lower_bounds > 0).sum(1) == K - 1)


    return Max_Q

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=16, help='Number of layers.')# APPNP10
parser.add_argument('--dims', type=list, default=[64], help='conv dims')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.') # APPNP2000 GCNII1500
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.001, help='weight decay (L2 loss on parameters).')# appnp 0.001 GCNII0.01
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')# appnp 0.5 GCNII 0.6
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
parser.add_argument('--Q-list', type=int, nargs="+", default=[1, 3, 5], help='Q values for certification')
parser.add_argument('--q', type=int, default=1, help='local feature perturbation budget')
parser.add_argument('--split', type=str, default="test", choices=["all", "train", "val", "test"], help='nodes to certify')
parser.add_argument('--output', type=str, default=None, help='csv output path')
parser.add_argument('--batch-size', type=int, default=1, help='certification batch size')
parser.add_argument('--only-correct', action="store_true", default=False, help='certify only correctly predicted nodes')
args = parser.parse_args()
# checkpt_file = 'pretrained/f596df8513aa455896d0a251eecf37fe.pt' ##128 cora
# checkpt_file = 'pretrained/9a1c2e6cf1364a8f8d5d7515f6e4dfd4.pt' ##128
# checkpt_file = 'pretrained/7d1742e20b9345da95e6427f321b78e1.pt' ##64
# checkpt_file = 'pretrained/e046415bc5ad446ca7dcea3f4a097da5.pt' ##32
default_citation_checkpoint = 'pretrained/48ccee97545b45f29c39161702dfe2d9.pt' ##16
# checkpt_file = 'pretrained/cd35a8a4fba44d6cb28ea899171d7497.pt' ##32 citeseer
# checkpt_file = 'pretrained/223fc6a516b74fa2a8bf3d7597124f8e.pt' ##16 pubmed
# checkpt_file = 'pretrained/7291b72ce75b4109aeb87fbd6bc22d1b.pt' ##8
# checkpt_file = 'pretrained/75b85d1015404f4cbc14daef4b726b24.pt' ## 10 APPNP cora

# dataset = torch_geometric.datasets.Planetoid(root='./dataset', name='Cora',transform = T.NormalizeFeatures())
# data = dataset[0]
cudaid = "cuda:" + str(args.dev)
device = torch.device(cudaid if torch.cuda.is_available() else "cpu")
print("Using device:", device)
data_key = args.data.lower()
if data_key in ["rcaeval-ob", "rcaeval_ob", "rcaeval"]:
    adj, features, labels, idx_train, idx_val, idx_test, a = load_rcaeval_ob()
    checkpt_file = args.checkpoint or 'pretrained/rcaeval_ob_gcnii_l{}.pt'.format(args.layer)
    model_adj = adj
    certify_A = a.to_dense() if a.is_sparse else a
elif data_key in ["rcaeval-ss", "rcaeval_ss"]:
    adj, features, labels, idx_train, idx_val, idx_test, a = load_rcaeval_ss()
    checkpt_file = args.checkpoint or 'pretrained/rcaeval_ss_gcnii_l{}.pt'.format(args.layer)
    model_adj = adj
    certify_A = a.to_dense() if a.is_sparse else a
elif data_key in ["rcaeval-tt", "rcaeval_tt"]:
    adj, features, labels, idx_train, idx_val, idx_test, a = load_rcaeval_tt()
    checkpt_file = args.checkpoint or 'pretrained/rcaeval_tt_gcnii_l{}.pt'.format(args.layer)
    model_adj = adj
    certify_A = a.to_dense() if a.is_sparse else a
else:
    adj, features, labels, idx_train, idx_val, idx_test, a = load_citation(args.data)
    checkpt_file = args.checkpoint or default_citation_checkpoint
    model_adj = a
    certify_A = a
print("Checkpoint:", checkpt_file)
# cudaid = "cuda:"+str(args.dev)
# device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
a = a.to(device)
model_adj = model_adj.to(device)
certify_A = certify_A.to(device)
# data = data.to(device)
model = RobustGRNModel(nfeat=features.shape[1],
                       adj=model_adj,
                nlayers=args.layer,
                # nhidden=64,
                dim=args.dims,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                lamda = args.lamda,
                alpha=args.alpha,
                variant=args.variant
                ).to(device)
# model = APPNP(nfeat=features.shape[1],
#               adj=adj,
#               K=10,
#               nhidden=64,
#               nclass=int(labels.max())+1,
#               dropout=args.dropout,
#               alpha=args.alpha).to(device)

model.load_state_dict(torch.load(checkpt_file, map_location=device))
model.eval()

if args.split == "train":
    nodes = idx_train
elif args.split == "val":
    nodes = idx_val
elif args.split == "test":
    nodes = idx_test
else:
    nodes = None

selected = torch.arange(features.shape[0]) if nodes is None else torch.as_tensor(nodes).long().cpu()
selected_before_correct_filter = len(selected)

with torch.no_grad():
    output = model(features, adj)
    pred = output.argmax(dim=1).cpu()
    labels_cpu = labels.cpu()

if args.only_correct:
    selected = selected[pred[selected] == labels_cpu[selected]]
    print("Only correctly predicted nodes: {} -> {}".format(selected_before_correct_filter, len(selected)))

selected_after_correct_filter = len(selected)
if selected_after_correct_filter == 0:
    raise ValueError("No nodes left to certify after applying split='{}' and only_correct={}".format(
        args.split,
        args.only_correct,
    ))

results = []

def summarize_group(prefix, mask, robust_selected, nonrobust_selected, neither_selected):
    group_num_nodes = int(mask.sum())
    group_robust_count = int(robust_selected[mask].sum())
    group_nonrobust_count = int(nonrobust_selected[mask].sum())
    group_neither_count = int(neither_selected[mask].sum())

    return {
        "{}_num_nodes".format(prefix): group_num_nodes,
        "{}_robust_count".format(prefix): group_robust_count,
        "{}_robust_percent".format(prefix): group_robust_count / group_num_nodes * 100 if group_num_nodes else 0.0,
        "{}_nonrobust_count".format(prefix): group_nonrobust_count,
        "{}_nonrobust_percent".format(prefix): group_nonrobust_count / group_num_nodes * 100 if group_num_nodes else 0.0,
        "{}_neither_count".format(prefix): group_neither_count,
        "{}_neither_percent".format(prefix): group_neither_count / group_num_nodes * 100 if group_num_nodes else 0.0,
    }

for Q_ in args.Q_list:
    print("Q:", Q_)
    start_time = time.time()
    robust_nodes, nonrobust_nodes = certify(model, features, args.q, A=certify_A, Q=Q_,
                          nodes=selected,
                          batch_size=args.batch_size,
                          certify_nonrobustness=True,
                          progress=True)
    runtime_seconds = time.time() - start_time

    selected_np = selected.numpy()
    robust_selected = robust_nodes[selected_np]
    nonrobust_selected = nonrobust_nodes[selected_np].astype(bool)
    neither_selected = (~robust_selected) & (~nonrobust_selected)
    selected_labels = labels[selected].cpu().numpy()
    root_mask = selected_labels == 1
    nonroot_mask = selected_labels == 0

    num_nodes = len(selected_np)
    robust_count = int(robust_selected.sum())
    nonrobust_count = int(nonrobust_selected.sum())
    neither_count = int(neither_selected.sum())
    root_result = summarize_group("root", root_mask, robust_selected, nonrobust_selected, neither_selected)
    nonroot_result = summarize_group("nonroot", nonroot_mask, robust_selected, nonrobust_selected, neither_selected)

    result = {
        "Q": int(Q_),
        "q": int(args.q),
        "split": args.split,
        "only_correct": bool(args.only_correct),
        "selected_before_correct_filter": int(selected_before_correct_filter),
        "selected_after_correct_filter": int(selected_after_correct_filter),
        "num_nodes": int(num_nodes),
        "robust_count": robust_count,
        "robust_percent": robust_count / num_nodes * 100 if num_nodes else 0.0,
        "nonrobust_count": nonrobust_count,
        "nonrobust_percent": nonrobust_count / num_nodes * 100 if num_nodes else 0.0,
        "neither_count": neither_count,
        "neither_percent": neither_count / num_nodes * 100 if num_nodes else 0.0,
        "runtime_seconds": runtime_seconds,
        "runtime_per_node": runtime_seconds / num_nodes if num_nodes else 0.0,
    }
    result.update(root_result)
    result.update(nonroot_result)
    results.append(result)

    print("Q={Q} q={q} split={split} num_nodes={num_nodes} robust={robust_count} ({robust_percent:.2f}%) "
          "nonrobust={nonrobust_count} ({nonrobust_percent:.2f}%) neither={neither_count} ({neither_percent:.2f}%) "
          "runtime={runtime_seconds:.2f}s per_node={runtime_per_node:.4f}s".format(**result))
    print("Root-cause: robust={root_robust_count} ({root_robust_percent:.2f}%) "
          "nonrobust={root_nonrobust_count} ({root_nonrobust_percent:.2f}%) "
          "neither={root_neither_count} ({root_neither_percent:.2f}%)".format(**result))
    print("Non-root: robust={nonroot_robust_count} ({nonroot_robust_percent:.2f}%) "
          "nonrobust={nonroot_nonrobust_count} ({nonroot_nonrobust_percent:.2f}%) "
          "neither={nonroot_neither_count} ({nonroot_neither_percent:.2f}%)".format(**result))

if args.output:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("Saved:", output_path)
