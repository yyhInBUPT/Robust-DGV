import math
import scipy as sp
import torch
import scipy.sparse as sp
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn import Linear
from model import *
# from torch_geometric.nn.conv import MessagePassing
from torch.nn.functional import relu
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def slice_adj(adj, nodes):
    # s = adj[nodes].nonzero()
    # node_l = []
    # for i in range(s.shape[0]):
    #     node_l.append(s[i][1].item())
    # nbs = np.unique(node_l)
    # adj_slice = adj[nodes][:, nbs].to(device)
    # return adj_slice, nbs
    if len(nodes) == adj.shape[0]:
        return adj, nodes
    s = adj[nodes].nonzero()
    nbs = torch.unique(s[:, 1]).to(device)

    adj_slice1 = adj[nodes].to(device)
    if len(nbs) == adj.shape[0]:
        return adj_slice1, nbs
    adj_slice = adj_slice1.index_select(1, nbs).to(device)
    # adj_slice = adj[nodes][:, nbs].to(device)
    return adj_slice, nbs

class APPNP(torch.nn.Module):
    def __init__(self, nfeat, adj, K, nhidden, nclass, dropout, alpha,):

        super(APPNP, self).__init__()
        self.dropout = dropout
        self.adj = adj
        self.K = K
        self.alpha = alpha
        self.nclass = nclass
        self.fcs = nn.ModuleList()
        self.convs = nn.ModuleList
        self.fcs.append(Robustlinear(nfeat, nhidden))
        self.fcs.append(Robustlinear(nhidden, nclass))
        self.params1 = list(self.fcs.parameters())
        # self.prop = APPNP_prop(K=K, alpha=alpha, args=args)
        # print(self.prop)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fcs[0](x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[1](x)
        hh=x
        for k in np.arange(self.K):
            x = self.adj.mm(x)
            x = (1-self.alpha)*x+self.alpha*hh
        # x = self.prop(x, adj)
        return F.log_softmax(x, dim=1)
    def predict(self, input, nodes, ):
        target = self.forward(input, ).max(-1)[1]
        return target[nodes]

    def get_neighbors(self, nodes,A):
        a = A[nodes].nonzero()
        node_l = []
        for i in range(a.shape[0]):
            node_l.append(a[i][1].item())

        return np.unique(node_l)
    def dual(self,phi,nodes,a,alpha,b,is_last=False):
        adj_slice,nbs = slice_adj(a,nodes)
        if is_last:
            next_phi=torch.einsum("ij,ilm->iljm", adj_slice, phi)
        else:
            next_phi = torch.einsum("ijkl,km->ijml", phi, adj_slice)
        w = (1-alpha)*torch.eye(next_phi.shape[3]).to(device)
        next_phi = torch.tensordot(next_phi, w, dims=((3,), (1,)))
        final_objective_term = next_phi.sum((-2, -1))
        bias = b.topk(1, dim=0)[0]
        bias_objective_term = (phi @ bias.t()).sum(2).squeeze()

        return next_phi,bias_objective_term,final_objective_term

    def dual_backward(self, input, nodes, q, Q, A, return_perturbations = False):
        input = input.float()
        batch = len(nodes)

        bounds = []

        fc_layer = self.fcs[0]
        lower_bound, upper_bound, input_layer = fc_layer.bounds_binary(input, q, Q)
        bounds.append((lower_bound, upper_bound))
        fc_layer = self.fcs[1]
        lower_bound, upper_bound, input_layer = fc_layer.bounds_continuous(input_layer, bounds[-1][0], bounds[-1][1])
        bounds.append((lower_bound, upper_bound))
        # for k in range(self.K-1):
        #     lower_bound = self.adj.mm(lower_bound)
        #     lower_bound = (1 - self.alpha) * lower_bound+self.alpha * input_layer
        #     upper_bound = self.adj.mm(upper_bound)
        #     upper_bound = (1 - self.alpha) * upper_bound + self.alpha * input_layer
        #     bounds.append((lower_bound, upper_bound))


        target_classes = self.predict(input, nodes)
        predicted_onehot = torch.eye(self.nclass)[target_classes]
        C_tensor = (predicted_onehot.unsqueeze(1) - torch.eye(self.nclass)).to(device)
        phis = [-C_tensor]

        bias_terms = torch.zeros([batch, self.nclass]).to(device)
        I_terms = torch.zeros([batch, self.nclass]).to(device)


        # phi = phis[-1]
        b = self.alpha*input_layer
        # w = (1-self.alpha)*torch.eye(input_layer.shape[0])
        # final = torch.zeros([batch, self.nclass]).to(device)

        # bias_term = phi @ b.t()
        # bias_terms += bias_term
        ## next_phi = final_objective_term
        for k in np.arange(1,self.K)[::-1]:
            phi = phis[-1]
            is_Last = k == self.K-1
            if k ==self.K-1:
                nbs = nodes
            else:
                nbs = self.get_neighbors(nbs,A)
            ret = self.dual(phi,nbs,A,self.alpha,b,is_Last)
            next_phi, bias_term, objective_term = ret
            phis.append(next_phi)
            bias_terms += bias_term
            if objective_term is not None:
                I_terms += objective_term
        # compute linear
        nbs = self.get_neighbors(nbs, A)
        ret = self.fcs[-1].dual_backward(phis[-1], bounds[0], nbs)
        next_phi, bias_term, objective_term = ret
        phis.append(next_phi)
        bias_terms += bias_term
        I_terms += objective_term
        phi_1_hat = self.fcs[0].phi_backward(phis[-1])
        bias_terms += self.fcs[0].bias_objective_term(phis[-1])

        Delta = relu(phi_1_hat).mul(1 - input[nbs]) + relu(-phi_1_hat).mul(input[nbs])

        q_largest_local, q_ixs = Delta.topk(q, dim=3)

        q_largest_overall = q_largest_local.reshape([batch, self.nclass, -1])
        n_sel = min(Q, q)
        Q_largest, Q_ixs = q_largest_overall.topk(n_sel, -1)

        rho = Q_largest[:, :, -1].unsqueeze(-1)

        # Indices of the perturbations
        if return_perturbations:
            q_ixs_reshape = q_ixs.reshape(batch, self.nclass, -1)
            Q_ixs_div = torch.div(Q_ixs, q, rounding_mode='trunc')
            pert_node_ixs = nbs[Q_ixs_div.cpu()]
            pert_dim_ixs = q_ixs_reshape.gather(-1, Q_ixs).cpu().numpy()
            perturbation_ixs = np.stack([pert_node_ixs, pert_dim_ixs], axis=-1)

        # Select the smallest of the q largest values per node,
        # or 0 if it is smaller than rho.
        # [B, K, Num L-1 hop neighbors]
        eta = relu(q_largest_local[:, :, :, -1] - rho)
        # Compute Psi (c.f. the paper) and sum over it
        # [B, K]
        Psi_term = relu(Delta - (rho + eta).unsqueeze(-1)).abs().sum((2, 3))
        # [B, K]
        trace_term = input[nbs].mul(phi_1_hat).sum((2, 3))

        # [B, K] lower-bound worst-case margins w.r.t. all other classes
        final_objective = I_terms - bias_terms - trace_term - Psi_term - q * eta.sum(-1) - Q * rho.squeeze(-1)

        if return_perturbations:
            return final_objective, perturbation_ixs

        return final_objective


