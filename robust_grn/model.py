import os

import math
import scipy as sp
import torch
import scipy.sparse as sp
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.functional import relu
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# dim卷积维数

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class GCNIILayer(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GCNIILayer, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

    def slice_adj(self,adj, nodes):
        if len(nodes) == adj.shape[0]:
            return adj, nodes
        s = adj[nodes].nonzero()
        nbs = torch.unique(s[:, 1]).to(device)

        adj_slice1 = adj[nodes].to(device)
        if len(nbs) == adj.shape[0]:
            return adj_slice1, nbs
        adj_slice = adj_slice1.index_select(1,nbs).to(device)
        # adj_slice = adj[nodes][:, nbs].to(device)
        return adj_slice, nbs


    def bounds_continuous(self,input,input_lower,input_upper,adj,h0,lamda,alpha,l,):
        ## 对偶中间边界计算

        theta = math.log(lamda/l+1)
        # hi = torch.spmm(adj,input)
        I_n = torch.eye(self.in_features).to(device)
        P = theta*self.weight+(1-theta)*I_n
        W = (1 - alpha) * P  ## output = adj*input.mul(W) + b
        b = alpha*h0.mm(P)

        W_plus = F.relu(W)
        W_minus = F.relu(-W)
        E = (input_lower > 0).float()
        I = ((input_lower < 0) & (input_upper > 0)).float()
        omega = (input_upper/(input_upper-input_lower + 1e-9)).mul(I)
        # omega_l = ((input_upper + input_lower) > 0).float().mul(I)
        ##考虑下界大于零以及下界小于零上界大于零的两类节点
        ll1 = input.mul(omega).mul(I)+input.mul(E)
        ll2 = input_lower.mul(omega)
        # ll3 = input.mul(omega_l).mul(I) + input.mul(E)
        # ll4 = input_lower.mul(omega_l)


        # lower_bound = adj.mm(ll1.mm(W)) + b - adj.mm(ll2.mm(W_minus))
        upper_bound = adj.mm(ll1.mm(W)) + b - adj.mm(ll2.mm(W_plus))
        lower_bound = adj.mm(ll1.mm(W)) + b - adj.mm(ll2.mm(W_minus))
        # lower_bound = adj.mm(ll3.mm(W)) + b - adj.mm(ll4.mm(W_minus))#

        input_layer = adj.mm(input.mm(W)) + b
        # lower_bound = adj @ (input_lower @ W_plus - input_upper @ W_minus) + b
        # upper_bound = adj @ (input_upper @ W_plus - input_lower @ W_minus) + b

        return lower_bound, upper_bound, input_layer

    def phi_backward(self, phi, adj, w, is_last=False):
        # torch.cuda.empty_cache()
        #[Num.L - l - 1 hop neighbors, Num L - l hop neighbors]

        if is_last:
            # phi: [Batch X Class X H^{L}] -> [Batch X Class X num L-1 neighbors X H^{L}]
            phi_hat = torch.einsum("ij,ilm->iljm", adj, phi)

        else:
            # phi: [Batch X Class X Num.L-l-1 neighbors X H^{l}] -> [Batch X Class X Num.L-l neighbors X H^{l}]
            phi_hat = torch.einsum("ijkl,km->ijml", phi, adj)

        # H^{l}->H^{l-1}
        phi_hat = torch.tensordot(phi_hat, w, dims=((3,), (1,)))

        return phi_hat

    def dual_backward(self, phi, nodes, bounds, A, h0,lamda,alpha,l,compute_objective=False,is_last=False):# , is_last=False
        # torch.cuda.empty_cache()
        adj_slice, nbs = self.slice_adj(A, nodes)
        lb, ub = bounds

        lb = lb[nbs]
        ub = ub[nbs]
        theta = math.log(lamda / l + 1)
        # hi = torch.spmm(adj,input)
        I_n = torch.eye(self.in_features).to(device)
        P = theta * self.weight + (1 - theta) * I_n
        W = (1 - alpha) * P  ## output = adj*input.mul(W) + b
        b = alpha * h0.mm(P)
        b = b[nodes]
        phi_hat = self.phi_backward(phi, adj_slice, W,is_last)# , is_last
        phi_hat_plus = relu(phi_hat)
        phi_hat_minus = relu(-phi_hat)

        omega = ub / (ub - lb + 1e-9)


        # consider the cases where the upper and lower bounds have different signs
        I = ((lb < 0) & (ub > 0)).float()
        I_plus = ((lb > 0) & (ub > 0)).float()

        phi_left = phi_hat.mul(I_plus)
        del I_plus
        phi_right_1 = phi_hat_plus.mul(omega)
        # phi_right_1 = phi_hat_plus.mul(ub / (ub - lb + 1e-9))
        phi_right_2 = phi_hat_minus.mul(omega)
        del phi_hat_minus
        phi_right = (phi_right_1 - phi_right_2).mul(I)
        del phi_right_1
        del phi_right_2

        # Phi l
        next_phi = phi_left + phi_right
        del phi_right
        del phi_left
        if compute_objective:
            final_objective_term = phi_hat_plus.mul(ub.mul(lb)/(ub - lb + 1e-9)).mul(I).sum((-2,-1))
        else:
            final_objective_term = None

        bias = b.topk(1, dim=0)[0]
        # bias = b.min(dim=0)[0]
        bias_objective_term = (phi@bias.t()).sum(2).squeeze()


        # bias_objective_term = (phi @ b.t()).sum(2)
        #
        # bias_objective_term = bias_objective_term.sum(2)
        # bias_objective_term = bias_objective_term.div(len(nbs))

        return next_phi, bias_objective_term, final_objective_term
    # def bias_objective_term(self,phi):
    #     b = self.bias
    #     phi = phi @ b
    #
    #     return (phi @ self.bias.unsqueeze(1)).sum(2).squeeze()



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
        W_plus = relu(W)
        W_minus = relu(-W)

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

        phi_hat_plus = relu(phi_hat)
        phi_hat_minus = relu(-phi_hat)

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




class RobustGRNModel(torch.nn.Module):
    def __init__(self, nfeat, adj, nlayers, dim, nclass, dropout, lamda, alpha, variant):
        super(RobustGRNModel, self).__init__()
        self.adj = adj
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GCNIILayer(dim[-1], dim[-1], variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(Robustlinear(nfeat, dim[0]))
        for i in range(len(dim)-1):
            self.fcs.append(Robustlinear(dim[i], dim[i+1]))
        self.fcs.append(Robustlinear(dim[-1], nclass))
        self.nclass = nclass
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i in range(1, len(self.fcs)-1):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(self.fcs[i](layer_inner))
            _layers.append(layer_inner)

        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[-1], self.lamda, self.alpha, i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

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
        # if not (torch.sort(input.unique().long().cpu())[0] == torch.tensor([0, 1])).all():
        #     raise ValueError("Node attributes must be binary.")
        # torch.cuda.empty_cache()
        input = input.float()
        batch = len(nodes)
        adj = self.adj

        bounds = []
        # compute bounds
        fc_layer = self.fcs[0]
        lower_bound, upper_bound, input_layer = fc_layer.bounds_binary(input, q, Q)
        bounds.append((lower_bound, upper_bound))
        ##计算中间层
        h0 = input_layer

        for i, layer in enumerate(self.convs):
            layer = self.convs[i]

            lower_bound, upper_bound, input_layer = layer.bounds_continuous(input_layer, bounds[-1][0], bounds[-1][1],
                                                                            adj, h0, self.lamda, self.alpha, i+1)
            bounds.append((lower_bound, upper_bound))
        # f_layer = self.fcs[-1]
        # lower_bound, upper_bound = f_layer.bounds_continuous(input_layer, bounds[-1][0], bounds[-1][-1])
        # bounds.append((lower_bound, upper_bound))
        del input_layer

        ##dual 计算

        target_classes = self.predict(input, nodes)
        # predicted_onehot = torch.eye(self.nclass)[target_classes]
        predicted_onehot = torch.eye(self.nclass, device=target_classes.device)[target_classes]
        C_tensor = predicted_onehot.unsqueeze(1) - torch.eye(self.nclass, device=predicted_onehot.device)
        phis = [-C_tensor]

        bias_terms = torch.zeros([batch, self.nclass]).to(device)
        I_terms = torch.zeros([batch, self.nclass]).to(device)
        # print(phis[0].device,I_terms.device)
        # final linear module computation
        # neighborhoods = self.get_neighborhoods(nodes, A)[::-1]

        w_l = self.fcs[-1].weight
        phi = phis[-1]
        phi = torch.einsum("ilm,mn->iln",phi,w_l)
        phis.append(phi)
        # ret = f_layer.dual_backward(phi, bounds[-1], nodes)# node
        # next_phi, bias_term, objective_term = ret
        # phis.append(next_phi)
        # bias_terms += bias_term
        # I_terms += objective_term


        #res module computation

        for i in np.arange(0, len(self.convs))[::-1]:
            # torch.cuda.empty_cache()
            layer = self.convs[i]
            phi = phis[-1]
            if i == len(self.convs)-1:
                nbs = nodes
            else:
                nbs = self.get_neighbors(nbs, A)

            compute_objective = i+1 > 0
            is_last_layer = i == len(self.convs) - 1

            ret = layer.dual_backward(phi, nbs, bounds[i], A, h0, self.lamda, self.alpha, i+1,
                                      compute_objective,is_last_layer)# node , is_last_layer
            next_phi, bias_term, objective_term = ret
            phis.append(next_phi)
            bias_terms += bias_term
            if objective_term is not None:
                I_terms += objective_term

        # nbs = neighborhoods[0]
        phi_1_hat = self.fcs[0].phi_backward(phis[-1])
        bias_terms += self.fcs[0].bias_objective_term(phis[-1])
        nbs = self.get_neighbors(nbs,A)
        input_hat = input[nbs]
        del ret
        del phis
        del bounds
        del A
        del adj
        del h0
        del next_phi
        del lower_bound
        del upper_bound
        del input
        del fc_layer


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
            del pert_dim_ixs
            del pert_node_ixs
            del Q_ixs_div
            del q_ixs_reshape

            # Select the smallest of the q largest values per node,
        # or 0 if it is smaller than rho.
        # [B, K, Num L-1 hop neighbors]
        eta = relu(q_largest_local[:, :, :, -1] - rho)
        # Compute Psi (c.f. the paper) and sum over it
        # [B, K]
        del q_largest_local
        del q_largest_overall
        del q_ixs
        del phi
        del nbs
        Psi_term = Delta-(rho+eta).unsqueeze(-1)
        del Delta
        Psi_term = relu(Psi_term).sum((2, 3))
        # Psi_term = relu(Psi_term).abs().sum((2, 3))


        # Psi_term = relu(Delta - (rho + eta).unsqueeze(-1)).abs().sum((2, 3))
        # [B, K]
        trace_term = input_hat.mul(phi_1_hat).sum((2, 3))
        del phi_1_hat

        # [B, K] lower-bound worst-case margins w.r.t. all other classes
        final_objective = I_terms - bias_terms - trace_term - Psi_term - q * eta.sum(-1) - Q * rho.squeeze(-1)

        if return_perturbations:
            return final_objective, perturbation_ixs

        return final_objective

class RobustGRNModel_PPI(torch.nn.Module):
    def __init__(self,nfeat, nlayers, dim, nclass, dropout, lamda, alpha, variant):
        super(RobustGRNModel_PPI, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GCNIILayer(dim[-1], dim[-1], variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(Robustlinear(nfeat, dim[0]))
        for i in range(len(dim)-1):
            self.fcs.append(Robustlinear(dim[i], dim[i+1]))
        self.fcs.append(Robustlinear(dim[-1], nclass))
        self.nclass = nclass
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i in range(1, len(self.fcs)-1):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(self.fcs[i](layer_inner))
            _layers.append(layer_inner)

        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[-1], self.lamda, self.alpha, i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        layer_inner = self.sig(layer_inner)
        # a=layer_inner
        return layer_inner

    def predict(self, input, nodes, adj):
        target = self.forward(input, adj).max(-1)[1]
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
    def sigma_dual_backward(self,):

        return

    def dual_backward(self, input, nodes, q, Q, adj, A, return_perturbations=False):
        # if not (torch.sort(input.unique().long().cpu())[0] == torch.tensor([0, 1])).all():
        #     raise ValueError("Node attributes must be binary.")
        # torch.cuda.empty_cache()
        input = input.float()
        batch = len(nodes)
        ##计算中间激活边界
        bounds = []
        fc_layer = self.fcs[0]
        lower_bound, upper_bound, input_layer = fc_layer.bounds_binary(input, q, Q)
        bounds.append((lower_bound, upper_bound))
        h0 = input_layer

        for i, layer in enumerate(self.convs):##ppi加一次
            layer = self.convs[i]

            lower_bound, upper_bound, input_layer = layer.bounds_continuous(input_layer, bounds[-1][0], bounds[-1][1],
                                                                            adj, h0, self.lamda, self.alpha, i+1)
            bounds.append((lower_bound, upper_bound))
        # f_layer = self.fcs[-1]
        # lower_bound, upper_bound = f_layer.bounds_continuous(input_layer, bounds[-1][0], bounds[-1][-1])
        # bounds.append((lower_bound, upper_bound))
        del input_layer



        ##dual 计算
        target_classes = self.predict(input, nodes, adj)
        # predicted_onehot = torch.eye(self.nclass)[target_classes]
        # C_tensor = (predicted_onehot.unsqueeze(1) - torch.eye(self.nclass)).to(device)
        predicted_onehot = torch.eye(self.nclass, device=target_classes.device)[target_classes]
        C_tensor = predicted_onehot.unsqueeze(1) - torch.eye(self.nclass, device=predicted_onehot.device)
        ##L->L-1
        phis = [-C_tensor]

        # bias_terms = torch.zeros([batch, self.nclass]).to(device)
        # I_terms = torch.zeros([batch, self.nclass]).to(device)
        bias_terms = torch.zeros([batch, self.nclass], device=predicted_onehot.device)
        I_terms = torch.zeros([batch, self.nclass], device=predicted_onehot.device)
        # print(phis[0].device,I_terms.device)
        # final linear module computation
        # neighborhoods = self.get_neighborhoods(nodes, A)[::-1]

        f_layer = self.fcs[-1]
        phi = phis[-1]
        ##L-1->L-2   sigmoid 计算
        ret = f_layer.dual_backward(phi, bounds[-1], nodes)# node
        next_phi, bias_term, objective_term = ret
        phis.append(next_phi)
        bias_terms += bias_term
        I_terms += objective_term

        #res module computation

        for i in np.arange(0, len(self.convs))[::-1]:
            # torch.cuda.empty_cache()
            layer = self.convs[i]
            phi = phis[-1]
            if i == len(self.convs)-1:
                nbs = nodes
            else:
                nbs = self.get_neighbors(nbs, A)

            compute_objective = i+1 > 0
            # is_last_layer = i == len(self.convs) - 1

            ret = layer.dual_backward(phi, nbs, bounds[i], A, h0, self.lamda, self.alpha, i+1,
                                      compute_objective)# node , is_last_layer
            next_phi, bias_term, objective_term = ret
            phis.append(next_phi)
            bias_terms += bias_term
            if objective_term is not None:
                I_terms += objective_term

        # nbs = neighborhoods[0]
        phi_1_hat = self.fcs[0].phi_backward(phis[-1], is_last=False)
        bias_terms += self.fcs[0].bias_objective_term(phis[-1])
        nbs = self.get_neighbors(nbs, A)
        input_hat = input[nbs]
        del ret
        del phis
        del bounds
        del A
        del adj
        del h0
        del next_phi
        del lower_bound
        del upper_bound
        del input
        del fc_layer


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
            del pert_dim_ixs
            del pert_node_ixs
            del Q_ixs_div
            del q_ixs_reshape

            # Select the smallest of the q largest values per node,
        # or 0 if it is smaller than rho.
        # [B, K, Num L-1 hop neighbors]
        eta = relu(q_largest_local[:, :, :, -1] - rho)
        # Compute Psi (c.f. the paper) and sum over it
        # [B, K]
        del q_largest_local
        del q_largest_overall
        del q_ixs
        del phi
        del nbs
        Psi_term = Delta-(rho+eta).unsqueeze(-1)
        del Delta
        Psi_term = relu(Psi_term).sum((2, 3))
        # Psi_term = relu(Psi_term).abs().sum((2, 3))


        # Psi_term = relu(Delta - (rho + eta).unsqueeze(-1)).abs().sum((2, 3))
        # [B, K]
        trace_term = input_hat.mul(phi_1_hat).sum((2, 3))
        del phi_1_hat

        # [B, K] lower-bound worst-case margins w.r.t. all other classes
        final_objective = I_terms - bias_terms - trace_term - Psi_term - q * eta.sum(-1) - Q * rho.squeeze(-1)

        if return_perturbations:
            return final_objective, perturbation_ixs

        return final_objective
