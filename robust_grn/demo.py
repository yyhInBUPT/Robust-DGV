import argparse

import torch
from matplotlib import pyplot as plt
from torch import optim

# from robust_grn.model import *
# from robust_grn.robust_appnp import *
from utils import load_citation
from tqdm import tqdm
import os
from process_nodes_classification import *
#from JKNet import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def chunker(seq, size):##块处理数据
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def certify(model, attrs,q, A, nodes=None, Q=12, optimize_steps=5,batch_size = 8,
           certify_nonrobustness=False, progress=True):

    N = attrs.shape[0]
    K = model.nclass


    if nodes == None:
        nodes = torch.arange(N)

    lower_bounds = []
    import time
    nonrobust_nodes = np.zeros(N)
    # batch = np.array([0,1,2,3,4,5,6,7])

    _iter = chunker(torch.arange(N), batch_size)
    if progress:
        _iter = tqdm(_iter, total=int(np.ceil(N / batch_size)))


    for chunk in _iter:
        lb, pert = model.dual_backward(attrs, chunk, q, Q, A,return_perturbations=True)# ,pert
        lb = lb.detach()
        lower_bounds.append(lb.cpu().numpy())
        if certify_nonrobustness:
            predicted_before = model.predict(attrs, chunk)
            predicted_after = []
            for ix, node in enumerate(chunk):
                attack_successful = []
                for cl in torch.sort(lb[ix])[1]:
                    if lb[ix, cl] >= 0:
                        # only test for nonrobustness when we cannot certify robustness
                        attack_successful.append(False)
                        continue

                    pert_ixs = tuple(torch.tensor(pert[ix, cl].T))
                    attrs.index_put_(pert_ixs, 1 - attrs[pert_ixs])
                    # nd = node.item()
                    # a = model.predict(attrs, [node]).cpu().numpy()

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
        # lower_bounds = np.row_stack(lower_bounds)
        # # print(lower_bounds[0])
        # robust_nodes = ((lower_bounds > 0).sum(1) == K - 1)



    lower_bounds = np.row_stack(lower_bounds)
    # print(lower_bounds[0])
    robust_nodes = ((lower_bounds > 0).sum(1) == K - 1)

    return robust_nodes, nonrobust_nodes

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=32, help='Number of layers.')# APPNP10
parser.add_argument('--dims', type=list, default=[64], help='conv dims')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.') # APPNP2000 GCNII1500
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.001, help='weight decay (L2 loss on parameters).')# appnp 0.001 GCNII0.01
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')# appnp 0.5 GCNII 0.6
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='citeseer', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
args = parser.parse_args()
# checkpt_file = 'pretrained/f596df8513aa455896d0a251eecf37fe.pt' ##128 cora
# checkpt_file = 'pretrained/9a1c2e6cf1364a8f8d5d7515f6e4dfd4.pt' ##128
# checkpt_file = 'pretrained/7d1742e20b9345da95e6427f321b78e1.pt' ##64
# checkpt_file = 'pretrained/e046415bc5ad446ca7dcea3f4a097da5.pt' ##32
# checkpt_file = 'pretrained/48ccee97545b45f29c39161702dfe2d9.pt' ##16
checkpt_file = 'pretrained/cd35a8a4fba44d6cb28ea899171d7497.pt' ##32 citeseer
# checkpt_file = 'pretrained/64122111f905ff4734aac4b9f47a2e51ed.pt' ##64
# checkpt_file = 'pretrained/128fa44ea0035db4cad99e71dd505ccf6e3.pt' ##128 73.1
# checkpt_file = 'pretrained/8daffbee71cad42f9aeb3b02238bbdce4.pt' ##8 71.9
# checkpt_file = 'pretrained/4ba67abf0a23d4e7083d68473027f8a76.pt' ##4 68.6
# checkpt_file = 'pretrained/275282390a8be4a87bb4395f378d1c4c5.pt' ##2 67.0
# checkpt_file = 'pretrained/f0714c8462d54891a43220fa3469e448.pt' ##16
# checkpt_file = 'pretrained/223fc6a516b74fa2a8bf3d7597124f8e.pt' ##16 pubmed
# checkpt_file = 'pretrained/7291b72ce75b4109aeb87fbd6bc22d1b.pt' ##8
# checkpt_file = 'pretrained/75b85d1015404f4cbc14daef4b726b24.pt' ## 10 APPNP cora
# checkpt_file = 'pretrained/5d69d48204b84c53b64987c5f171861a.pt'##16 Actor acc 33.9
# checkpt_file = 'pretrained/f4ad66882a8b4d20a25e775355395304.pt'##8 35.1
# checkpt_file = 'pretrained/28e9dbb4c5b44840a1dd06d656111546.pt'## 8 Amazon_photo 94.9
# checkpt_file = 'pretrained/f4861d6ab92c4be8a2c41d245a343e7b.pt'## 16 94.6
# checkpt_file = 'pretrained/8aa3d4ed1ae24827bfa1f5c75807bdab.pt'## 16 Amazno_cs 84.2
# checkpt_file = 'pretrained/d69b6616a62240ea978049ef476db630.pt'## 8 88.1
# checkpt_file = 'pretrained/777dba3506b141b6802ba5c95d9f921eppi.pt'
# checkpt_file = 'pretrained/JK'
# dataset = torch_geometric.datasets.Planetoid(root='./dataset', name='Cora',transform = T.NormalizeFeatures())
# data = dataset[0]
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# adj, features, labels,idx_train,idx_val,idx_test,a = process_Actor()
# adj, features, labels, idx_train, idx_val, idx_test, a = process_Amazon_CS()
adj, features, labels, idx_train, idx_val, idx_test, a = load_citation(args.data)
# cudaid = "cuda:"+str(args.dev)
# device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
a = a.to(device)


# data = data.to(device)
model = RobustGRNModel(nfeat=features.shape[1],
                       adj=a,##a dense;adj sparse
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
# model = JKNet(nfeat = features.shape[1], adj=adj,nclass=int(labels.max())+1,mode='cat', num_layers=6, hidden=16).to(device)
model.load_state_dict(torch.load(checkpt_file,map_location={'cuda:0': 'cuda:1'}))


# q = 5
q = int(0.01*features.shape[1])
# Q=12
batch_size = 8
accruracy= []
# Q_range = [1,2,3,4,5,6,7,8,9,10]# 1 5 12 15 20#5,10,15,,35,50 5,10,15,25,35,50
# Q_range = [1,2,3,4,5]
# Q_range=[1,10,20,40,60,80,100]
# Q_range = [1,20,40,50,85,95,105,110,115]
# Q_range=[5,10,15,30,45,60,75,90,100,120]
Q_range=[4,6,8,10,12]
# for i in range(16):
#     Q_range.append(i+1)
certifiable_nodes = []
i=0
for Q_ in Q_range:
    print(Q_)
    certifiable_nodes.append(certify(model, features, q, A=a,Q=Q_,
                          batch_size=batch_size,
                          certify_nonrobustness=True,
                          progress=True))
    print(np.round((certifiable_nodes[i][0].mean()) * 100, 2))
    print(np.round((certifiable_nodes[i][1].mean()) * 100, 2))
    i = i+1
# np.save('Res_Amazon_CS_Q_55.npy', certifiable_nodes)
    # robust = np.array([x[0] for x in certifiable_nodes])
    # nonrobust = np.array([x[1] for x in certifiable_nodes])
    # print(robust, nonrobust)
robust = np.array([x[0] for x in certifiable_nodes])
nonrobust = np.array([x[1] for x in certifiable_nodes])

print(robust, nonrobust)

cer = 100 * (robust.mean(1))
ncer = 100 * (1 - nonrobust.mean(1))
plt.plot(Q_range, 100*(robust.mean(1)), linewidth=3)
plt.fill_between(Q_range, 0, 100*robust.mean(1), alpha=0.2, label="Certifiably robust")
plt.plot(Q_range, 100*(1-nonrobust.mean(1)), linewidth=3)
plt.fill_between(Q_range, 100, 100*(1-nonrobust.mean(1)),
                alpha=0.2, label="Certifiably nonrobust")
plt.xlabel("Number of perturbations", fontsize=14)
plt.ylabel("% certifiable nodes", fontsize=14)

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=14)
plt.ylim((-3,103))
plt.legend(fontsize=14)
plt.show()
    # print(np.round(robust_at_Q*100, 2))
    # accruracy.append(np.round(robust_at_Q*100, 2))
# print(accruracy)

