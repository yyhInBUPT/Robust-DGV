import os.path as osp
import argparse
import sys
import time
import uuid

import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_networkx

# import torch_geometric
#
# import torch_geometric.transforms as T

from model import *
from robust_appnp import APPNP
import math
from torch_geometric.datasets import Actor

from robust_grn.process_nodes_classification import process_Actor
from utils import *
from process_nodes_classification import *

from sklearn.metrics import precision_recall_fscore_support, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')# APPNP10
parser.add_argument('--dims', type=list, default=[64], help='conv dims')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.') # APPNP2000 GCNII1500
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')# appnp 0.001 GCNII0.01
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')# appnp 0.5 GCNII 0.6
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='citeseer', help='dateset')
parser.add_argument('--dev', type=int, default=1, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
args = parser.parse_args()
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# dataset = torch_geometric.datasets.Planetoid(root='./dataset', name='Pubmed', transform = T.NormalizeFeatures())
# data = dataset[0]

# print(data.train_mask.sum())
# print(data.val_mask.sum())
# print(data.test_mask.sum())

data_key = args.data.lower()

if args.checkpoint is not None:
    checkpt_file = args.checkpoint
elif data_key in ["rcaeval-ob", "rcaeval_ob", "rcaeval"]:
    checkpt_file = 'pretrained/rcaeval_ob_gcnii_l{}.pt'.format(args.layer)
elif data_key in ["rcaeval-ss", "rcaeval_ss"]:
    checkpt_file = 'pretrained/rcaeval_ss_gcnii_l{}.pt'.format(args.layer)
elif data_key in ["rcaeval-tt", "rcaeval_tt"]:
    checkpt_file = 'pretrained/rcaeval_tt_gcnii_l{}.pt'.format(args.layer)
else:
    checkpt_file = 'pretrained/'+'2'+uuid.uuid4().hex+'.pt'
print("Checkpoint:", checkpt_file)
# checkpt_file = 'pretrained/f596df8513aa455896d0a251eecf37fe.pt' ##128 cora
# checkpt_file = 'pretrained/9a1c2e6cf1364a8f8d5d7515f6e4dfd4.pt' ##128
# checkpt_file = 'pretrained/7d1742e20b9345da95e6427f321b78e1.pt' ##64
# checkpt_file = 'pretrained/e046415bc5ad446ca7dcea3f4a097da5.pt' ##32
# checkpt_file = 'pretrained/48ccee97545b45f29c39161702dfe2d9.pt' ##16
# checkpt_file = 'pretrained/cd35a8a4fba44d6cb28ea899171d7497.pt' ##32 citeseer 128 73.1
# checkpt_file = 'pretrained/223fc6a516b74fa2a8bf3d7597124f8e.pt' ##16 pubmed
# adj, features, labels,idx_train,idx_val,idx_test,a = process_Amazon_CS()
# adj, features, labels,idx_train,idx_val,idx_test,a = process_Actor()
# adj, features, labels,idx_train,idx_val,idx_test,a = load_citation(args.data)
if data_key in ["rcaeval-ob", "rcaeval_ob", "rcaeval"]:
    adj, features, labels, idx_train, idx_val, idx_test, a = load_rcaeval_ob()
elif data_key in ["rcaeval-ss", "rcaeval_ss"]:
    adj, features, labels, idx_train, idx_val, idx_test, a = load_rcaeval_ss()
elif data_key in ["rcaeval-tt", "rcaeval_tt"]:
    adj, features, labels, idx_train, idx_val, idx_test, a = load_rcaeval_tt()
else:
    adj, features, labels, idx_train, idx_val, idx_test, a = load_citation(args.data)

features = features.to(device)
adj = adj.to(device)
# # data = data.to(device)

model = RobustGRNModel(nfeat=features.shape[1],
                       adj=adj,
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

optimizer = torch.optim.Adam([
    dict(params=model.params1, weight_decay=args.wd1),
    dict(params=model.params2, weight_decay=args.wd2)
], lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)#GCNII
    # output = model(features)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    if data_key in ["rcaeval-ob", "rcaeval_ob", "rcaeval", "rcaeval-ss", "rcaeval_ss", "rcaeval-tt", "rcaeval_tt"]:
        train_labels = labels[idx_train]
        class_count = torch.bincount(train_labels, minlength=int(labels.max()) + 1).float()
        class_weight = class_count.sum() / (len(class_count) * class_count)
        class_weight = class_weight.to(device)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device), weight=class_weight)
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features,adj)#GCNII
        # output = model(features)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()


def test():
    model.load_state_dict(torch.load(checkpt_file, map_location=device))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))

        y_true = labels[idx_test].cpu().numpy()
        y_pred = output[idx_test].argmax(dim=1).cpu().numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        binary_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        print("Test label distribution:", {int(i): int((y_true == i).sum()) for i in set(y_true)})
        print("Pred label distribution:", {int(i): int((y_pred == i).sum()) for i in set(y_pred)})
        print("Class 0 Precision/Recall/F1: {:.4f} {:.4f} {:.4f}".format(precision[0], recall[0], f1[0]))
        print("Class 1 Precision/Recall/F1: {:.4f} {:.4f} {:.4f}".format(precision[1], recall[1], f1[1]))
        print("Macro-F1: {:.4f}".format(macro_f1))
        print("Root-cause F1: {:.4f}".format(binary_f1))

        return loss_test.item(), acc_test.item()

# model.load_state_dict(torch.load(checkpt_file))
# model.eval()
t_total = time.time()
bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
for epoch in range(args.epochs):
    loss_tra, acc_tra = train()
    loss_val, acc_val = validate()
    if (epoch + 1) % 1 == 0:
        print('Epoch:{:04d}'.format(epoch + 1),
              'train',
              'loss:{:.3f}'.format(loss_tra),
              'acc:{:.2f}'.format(acc_tra * 100),
              '| val',
              'loss:{:.3f}'.format(loss_val),
              'acc:{:.2f}'.format(acc_val * 100))
    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

if args.test:
    acc = test()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test" if args.test else "Val", "acc.:{:.1f}".format(acc * 100))
