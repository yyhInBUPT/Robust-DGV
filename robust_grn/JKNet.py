import uuid

import math
from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import JumpingKnowledge
from torch.nn.parameter import Parameter
from torch.nn.functional import relu

from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RobustConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(RobustConv,self).__init__()
        self.x = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.bias = Parameter(torch.empty(self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.out_features)
        # self.weight.data.uniform_(-stdv, stdv)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
    def forward(self,input,adj):
        W = self.weight.t().to(device)
        output = adj.mm(input.mm(W))+self.bias
        return output

    def phi_back(self,phi,adj,is_last):
        if is_last:

            phi_hat = torch.einsum("ij,ilm->iljm", adj, phi)

        else:

            phi_hat = torch.einsum("ijkl,km->ijml", phi, adj)

        # H^{l}->H^{l-1}
        phi_hat = torch.tensordot(phi_hat, self.weight, dims=((3,), (1,)))

        return phi_hat

    def dual_backward(self, phi, nodes, bounds, A,compute_objective=False,is_last=False):
        adj_slice, nbs = self.slice_adj(A, nodes)
        lb, ub = bounds
        lb = lb[nodes]
        ub = ub[nodes]

        phi_hat = self.phi_backward(phi,adj_slice, is_last)

        phi_hat_plus = relu(phi_hat)
        phi_hat_minus = relu(-phi_hat)

        omega = ub / (ub - lb + 1e-9)

        I = ((lb < 0) & (ub > 0)).float()
        I_plus = ((lb > 0) & (ub > 0)).float()

        phi_left = phi_hat.mul(I_plus)
        phi_right_1 = phi_hat_plus.mul(ub / (ub - lb + 1e-9))
        phi_right_2 = phi_hat_minus.mul(omega)
        phi_right = (phi_right_1 - phi_right_2).mul(I)

        # Phi l
        next_phi = phi_left + phi_right

        if compute_objective:
            final_objective_term = phi_hat_plus.mul(ub.mul(lb)/(ub - lb + 1e-9)).mul(I).sum((-2,-1))
        else:
            final_objective_term = None

        bias_objective_term = (phi @ self.bias.unsqueeze(1)).sum(2).squeeze()

        return next_phi, bias_objective_term, final_objective_term

    def bias_objective_term(self, phi):
        return (phi @ self.bias.unsqueeze(1)).sum(2).squeeze()

    def bounds_binary(self, input, q, Q, batch_size=8):
        input_extended = input.unsqueeze(2)
        W = self.weight.t()
        W_plus = F.relu(W).unsqueeze(0)
        W_minus = F.relu(-W).unsqueeze(0)

        lower_item = input_extended.mul(W_plus) + (1 - input_extended).mul(W_minus)
        lower_top_q_vals = lower_item.topk(q, 1)[0]
        lower_top_q_vals = lower_top_q_vals.reshape([-1, self.out_features])

        lower_top_Q_vals = lower_top_q_vals.topk(k=Q, dim=0)[0].sum(0)
        lower_bound = self.forward(input) - lower_top_Q_vals

        upper_item = (1 - input_extended).mul(W_plus) + input_extended.mul(W_minus)
        upper_top_q_vals = upper_item.topk(q, 1)[0]

        upper_top_q_vals = upper_top_q_vals.reshape([-1, self.out_features])
        upper_top_Q_vals = upper_top_q_vals.topk(k=Q, dim=0)[0].sum(0)

        upper_bound = self.forward(input) + upper_top_Q_vals

        input_layer = self.forward(input)
        input_layers=[]
        input_layers.append(input_layer)

        return lower_bound, upper_bound, input_layers

    def bounds_continuous(self, input_layers, input_lower, input_upper,is_last,  ):
        input_layer = self.forward(input_layers[-1])
        # input_layers.append(input_layer)
        # torch.cuda.empty_cache()

        E = (input_lower > 0).float()
        I = ((input_upper > 0) & (input_lower < 0)).float()
        W = self.weight.t()
        W_plus = relu(W)
        W_minus = relu(-W)

        omega = ((input_upper / (input_upper - input_lower + 1e-9)).mul(I))
        # omega_l = ((input_upper + input_lower) > 0).float().mul(I)

        ll1 = input.mul(omega).mul(I) + input.mul(E)
        ll2 = input_lower.mul(omega)

        #
        # ll3=input.mul(omega_l).mul(I)+input.mul(E)
        # ll4=input_lower.mul(omega_l)
        #
        lower_bound = self.forward(ll1) - ll2.mm(W_minus)
        upper_bound = self.forward(ll1) - ll2.mm(W_plus)
        if is_last:
            upper_bounds = input_layers
            lower_bounds = input_layers
            upper_bounds.append(upper_bound)
            lower_bounds.append(lower_bound)
            lower_bound = torch.cat(lower_bounds, dim=1)
            upper_bound = torch.cat(upper_bounds, dim=1)
        else:
            input_layers.append(input_layer)

        # lower_bound = self.forward(ll3) - ll4.mm(W_minus)


        return lower_bound, upper_bound, input_layers


class Robustlinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Robustlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    def bounds_binary(self, input, q, Q,batch_size=8):
        N = input.shape[0]

        input_extended = input.unsqueeze(2)
        W = self.weight.t()
        W_plus = F.relu(W).unsqueeze(0)
        W_minus = F.relu(-W).unsqueeze(0)
        # adj = A.unsqueeze(2)



        lower_item = input_extended.mul(W_plus) + (1 - input_extended).mul(W_minus)
        # lower_item = W_plus.mul(input_extended) + W_minus.mul(1-input_extended)
        # lower_top = lower_item.topk(q,1)

        lower_top_q_vals = lower_item.topk(q, 1)[0]  #[N, D, H]->[N, q, H] 提出每个节点特征的最大q
        # lower_top_q_vals = lower_top_q_vals.reshape([N, -1])
        # lower_top_q_vals = adj.mul(lower_top_q_vals).reshape([N, N, q, -1])#直接乘矩阵变成N N q H
        # lower_top_q_vals = lower_top_q_vals.unsqueeze(2)
        # lower_top_q_vals = lower_top_q_vals.repeat_interleave(batch_size, dim=2)
        lower_top_q_vals = lower_top_q_vals.reshape([-1,self.out_features])# [N*q H]

        # n_sel = min(Q, N*q) ##Q nbs*q
        # lower_top_Q_vals = lower_top_q_vals.reshape([N, -1, self.out_features]).topk(k=n_sel, dim=1)[0].sum(1)
        lower_top_Q_vals = lower_top_q_vals.topk(k=Q,dim=0)[0].sum(0)
        lower_bound = self.forward(input) - lower_top_Q_vals
        del lower_top_q_vals
        del lower_item
        del lower_top_Q_vals



        upper_item = (1 - input_extended).mul(W_plus) + input_extended.mul(W_minus)
        upper_top_q_vals = upper_item.topk(q, 1)[0]
        del upper_item
        upper_top_q_vals = upper_top_q_vals.reshape([-1,self.out_features])
        upper_top_Q_vals = upper_top_q_vals.topk(k=Q,dim=0)[0].sum(0)
        # upper_top_q_vals = upper_top_q_vals.unsqueeze(2)
        # upper_top_q_vals = upper_top_q_vals.repeat_interleave(batch_size, dim=2)
        # upper_top_q_vals = upper_top_q_vals.reshape([N,-1])
        # upper_top_q_vals = adj.mul(upper_top_q_vals).reshape([N,N,q,-1])

        # upper_top_Q_vals = upper_top_q_vals.reshape([N, -1, self.out_features]).topk(k=n_sel, dim=1)[0].sum(1)
        upper_bound = self.forward(input) + upper_top_Q_vals
        del upper_top_q_vals

        del upper_top_Q_vals
        input_layer = self.forward(input)

        return lower_bound, upper_bound, input_layer

    def bounds_continuous(self, input, input_lower, input_upper, ):
        input_layer = self.forward(input)
        # torch.cuda.empty_cache()

        E = (input_lower > 0).float()
        I = ((input_upper > 0) & (input_lower < 0)).float()
        W = self.weight.t()
        W_plus = F.relu(W)
        W_minus = F.relu(-W)

        omega = ((input_upper / (input_upper - input_lower + 1e-9)).mul(I))
        omega_l = ((input_upper + input_lower) > 0).float().mul(I)

        ll1 = input.mul(omega).mul(I) + input.mul(E)
        ll2 = input_lower.mul(omega)

        ll3=input.mul(omega_l).mul(I)+input.mul(E)
        ll4=input_lower.mul(omega_l)

        # lower_bound = self.forward(ll1) - ll2.mm(W_minus)
        upper_bound = self.forward(ll1) - ll2.mm(W_plus)
        lower_bound = self.forward(ll3) - ll4.mm(W_minus)


        return lower_bound, upper_bound, input_layer

    def phi_backward(self, phi):#nodes, dim,is_last

        # adj = torch.eye(dim).to(device)
        W = self.weight.t()

        # print(adj.device, phi.device)
        # if is_last:
        #     # [...,H^{L}]->[...,H^{L-1}]
        #     A = torch.eye(phi.shape[0]).to(device)
        #     phi_hat = torch.einsum("ij,ilm->iljm",A,phi)
        #     # phi_hat = phi.unsqueeze(2)
        #     # phi_hat = phi_hat.repeat_interleave(phi.shape[0], dim=2)
        #     phi_hat = torch.einsum("ijkl,ml->ijkm",phi_hat,W)
        #     # phi_hat = torch.einsum("ilm,km->ilk",phi,W)
        #
        # else:
        #     # A = torch.eye(phi.shape[2]).to(device)
        #     # phi_hat = torch.einsum("ijkl,km->ijml",phi,A)
        #
        #     phi_hat = torch.einsum("ijkl,ml->ijkm",phi,W)
        #     # phi_hat = torch.einsum("ijkl,km->ijml", phi, adj)
        phi_hat = torch.einsum("ijkl,ml->ijkm", phi, W)



        return phi_hat


    def dual_backward(self, phi, bounds, nodes):
        # torch.cuda.empty_cache()
        lb, ub = bounds
        lb = lb[nodes]
        ub = ub[nodes]
        # W = self.weight.t()
        # phi_hat = torch.einsum("ijl,ml->ijm", phi, W)

        phi_hat = self.phi_backward(phi)

        phi_hat_plus = F.relu(phi_hat)
        phi_hat_minus = F.relu(-phi_hat)

        omega = ub / (ub - lb + 1e-9)

        # consider the cases where the upper and lower bounds have different signs
        I = ((lb < 0) & (ub > 0)).float()
        I_plus = ((lb > 0) & (ub > 0)).float()

        phi_left = phi_hat.mul(I_plus)
        phi_right_1 = phi_hat_plus.mul(ub / (ub - lb + 1e-9))
        phi_right_2 = phi_hat_minus.mul(omega)
        phi_right = (phi_right_1 - phi_right_2).mul(I)

        # Phi l
        next_phi = phi_left + phi_right

        final_objective_term = phi_hat_plus.mul(ub.mul(lb) / (ub - lb + 1e-9)).mul(I).sum((-2, -1))

        bias_objective_term = (phi @ self.bias.unsqueeze(1)).sum(2).squeeze()


        return next_phi, bias_objective_term,final_objective_term
    def bias_objective_term(self,phi):
        return (phi @ self.bias.unsqueeze(1)).sum(2).squeeze()

class JKNet(nn.Module):
    def __init__(self, nfeat, adj, nclass, mode='cat', num_layers=6, hidden=16):
        super(JKNet, self).__init__()
        self.num_layers = num_layers
        self.mode = mode
        self.adj = adj
        self.conv0 = RobustConv(nfeat, hidden)
        self.dropout0 = nn.Dropout(p=0.5)

        for i in range(1, self.num_layers):
            setattr(self, 'conv{}'.format(i), RobustConv(hidden, hidden))
            setattr(self, 'dropout{}'.format(i), nn.Dropout(p=0.5))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = Robustlinear(hidden, nclass)
        elif mode == 'cat':
            self.fc = Robustlinear(num_layers * hidden, nclass)

    def forward(self, nfeat,adj):
        # x, edge_index = data.x, data.edge_index
        x = nfeat

        layer_out = []  # 保存每一层的结果
        for i in range(self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            dropout = getattr(self, 'dropout{}'.format(i))
            x = dropout(F.relu(conv(x, adj)))
            layer_out.append(x)

        h = self.jk(layer_out)  # JK层

        h = self.fc(h)
        h = F.log_softmax(h, dim=1)

        return h
    def predict(self, input, nodes, ):
        target = self.forward(input, self.adj).max(-1)[1]
        return target[nodes]

    def get_neighbors(self, nodes,A):
        if len(nodes)==A.shape[0]:
            return nodes
        # a = A[nodes].nonzero()
        # node_l = []
        # for i in range(a.shape[0]):
        #     node_l.append(a[i][1].item())
        #
        # return np.unique(node_l)
        neighbors = A[nodes].nonzero()
        neighbors = torch.unique(neighbors[:,1])
        return neighbors

    def dual_backward(self, input, nodes, q, Q, A, return_perturbations=False):
        input = input.float()
        batch = len(nodes)
        #adj = self.adj

        bounds = []
        fl = self.gconv0
        lower_bound, upper_bound, input_layers = fl.bounds_binary(input, q, Q)
        bounds.append(lower_bound, upper_bound)

        for i, layer in enumerate(1, self.n_layers):
            is_last = i == self.n_layers - 1
            layer = getattr(self, 'gconv{}'.format(i))
            lower_bound, upper_bound, input_layers = layer.bounds_continous(input_layers, bounds[-1][0],
                                                                            bounds[-1][1], is_last)
            bounds.append((lower_bound, upper_bound))

        target_classes = self.predict(input, nodes)
        predicted_onehot = torch.eye(self.nclass)[target_classes]
        c_tensor = (predicted_onehot.unsqueeze(1) - torch.eye(self.nclass)).to(device)
        phis = [-c_tensor]

        bias_terms = torch.zeros([batch, self.nclass]).to(device)
        I_terms = torch.zeros([batch, self.nclass]).to(device)

        w_l = self.fc.weight
        phi = phis[-1]
        phi = torch.einsum("ilm,mn->iln", phi, w_l)
        phis.append(phi)


        for i in np.arange(self.n_layers)[::-1]:
            layer = getattr(self, 'gconv{}'.format(i))
            phi = phis[-1]
            if i == self.n_layers - 1:
                nbs = nodes
            else:
                nbs = self.get_neighbors(nbs, A)
            compute_objective = i + 1 > 0
            is_last_layer = i == len(self.convs) - 1
            ret = layer.dual_backward(phi,nbs,bounds[i],A,compute_objective,is_last_layer)
            next_phi, bias_term, objective_term = ret
            phis.append(next_phi)
            bias_terms += bias_term
            if objective_term is not None:
                I_terms += objective_term

        phi_1_hat = self.gconv0.phi_back(phis[-1])
        bias_terms += self.gconv0.bias_objective_term(phis[-1])
        nbs = self.get_neighbors(nbs, A)
        input_hat = input[nbs]

        Delta = relu(phi_1_hat).mul(1 - input_hat) + relu(-phi_1_hat).mul(input_hat)

        q_largest_local, q_ixs = Delta.topk(q, dim=3)

        q_largest_overall = q_largest_local.reshape([batch, self.nclass, -1])
        n_sel = min(Q, q)
        Q_largest, Q_ixs = q_largest_overall.topk(n_sel, -1)

        rho = Q_largest[:, :, -1].unsqueeze(-1)

        # Indices of the perturbations
        if return_perturbations:
            q_ixs_reshape = q_ixs.reshape(batch, self.nclass, -1)
            Q_ixs_div = torch.div(Q_ixs, q, rounding_mode='trunc')
            pert_node_ixs = nbs[Q_ixs_div.cpu()].cpu().numpy()
            pert_dim_ixs = q_ixs_reshape.gather(-1, Q_ixs).cpu().numpy()
            perturbation_ixs = np.stack([pert_node_ixs, pert_dim_ixs], axis=-1)

        eta = relu(q_largest_local[:, :, :, -1] - rho)
        Psi_term = Delta - (rho + eta).unsqueeze(-1)
        Psi_term = relu(Psi_term).sum((2, 3))
        trace_term = input_hat.mul(phi_1_hat).sum((2, 3))
        final_objective = I_terms - bias_terms - trace_term - Psi_term - q * eta.sum(-1) - Q * rho.squeeze(-1)

        if return_perturbations:
            return final_objective, perturbation_ixs
        return final_objective


adj, features, labels, idx_train, idx_val, idx_test, a = load_citation("cora")
features = features.to(device)
adj = adj.to(device)
a = a.to(device)
labels = labels.to(device)
model = JKNet(nfeat=features.shape[1], adj=adj, nclass=int(labels.max()) + 1, mode='cat')  # max和cat两种模式可供选择
# model = GCNNet(dataset)
# model = GATNet(dataset)
model = model.to(device)
print(model)



# data = dataset[0].to(device)
# print(data)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)#5e-4



def train():
    model.train()
    for epoch in range(1000):
        out = model(nfeat=features, adj=adj)
        loss = criterion(out[idx_train], labels[idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out[idx_train], dim=1)
        correct = (pred == labels[idx_train]).sum().item()

        acc = correct / idx_train.shape[0]

        print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
            epoch, loss.item(), acc))

        # val_loss, val_acc = valid()

        # print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f} val_loss: {:.4f} val_acc: {:.4f}'.format(
        #     epoch, loss.item(), acc, val_loss, val_acc))

    test()
    checkpt_file = 'pretrained/' +'JK'+ uuid.uuid4().hex + '.pt'
    torch.save(model.state_dict(), checkpt_file)


# def valid():
#     # model.eval()
#     with torch.no_grad():
#         out = model(data)
#         loss = criterion(out[data.val_mask], data.y[data.val_mask])
#         _, pred = torch.max(out[data.val_mask], dim=1)
#         correct = (pred == data.y[data.val_mask]).sum().item()
#         acc = correct / data.val_mask.sum().item()
#         return loss.item(), acc
#         # print("val_loss: {:.4f} val_acc: {:.4f}".format(loss.item(), acc))


def test():
    model.eval()
    out = model(nfeat=features, adj=adj)
    loss = criterion(out[idx_test], labels[idx_test])
    _, pred = torch.max(out[idx_test], dim=1)
    correct = (pred == labels[idx_test]).sum().item()
    acc = correct / idx_test.shape[0]
    print("test_loss: {:.4f} test_acc: {:.4f}".format(loss.item(), acc))
def certify():
    return

if __name__ == '__main__':
    train()
