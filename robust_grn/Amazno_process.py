import torch
import os
import os.path as osp
from scipy.sparse import csr_matrix
from utils import *

def load_npz(filepath):
    filepath = osp.abspath(osp.expanduser(filepath))

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    if osp.isfile(filepath):
        with np.load(filepath, allow_pickle=True) as loader:
            loader = dict(loader)
            for k, v in loader.items():
                if v.dtype.kind in {'O', 'U'}:
                    loader[k] = v.tolist()
            return loader
    else:
        raise ValueError(f"{filepath} doesn't exist.")
def process_Amazon_Photo():
    dataset = load_npz('amazon_electronics_photo.npz')
    labels = dataset['labels']
    labels = torch.tensor(labels, dtype=torch.int64)
    attr_indptr = dataset['attr_indptr']
    attr_indices = dataset['attr_indices']
    attr_data = dataset['attr_data']
    # 将 attr_indptr、attr_indices、attr_data 转换为 Tensor
    attr_indptr = torch.tensor(attr_indptr, dtype=torch.int64)
    attr_indices = torch.tensor(attr_indices, dtype=torch.int64)
    attr_data = torch.tensor(attr_data, dtype=torch.float32)

    num_nodes = len(attr_indptr) - 1
    num_features = max(attr_indices) + 1

    # 创建稀疏节点特征矩阵
    row_indices = torch.repeat_interleave(torch.arange(num_nodes), attr_indptr[1:] - attr_indptr[:-1])
    col_indices = attr_indices
    features = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]),
        attr_data,
        size=(num_nodes, num_features)
    )
    features = features.to_dense()
    # 特征矩阵
    adj_indptr = dataset['adj_indptr']
    adj_indices = dataset['adj_indices']
    adj_data = dataset['adj_data']
    # 将 adj_indptr、adj_indices、adj_data 转换为 Tensor
    adj_indptr = torch.tensor(adj_indptr, dtype=torch.int64)
    adj_indices = torch.tensor(adj_indices, dtype=torch.int64)
    adj_data = torch.tensor(adj_data, dtype=torch.float32)
    # 确定邻接矩阵的形状
    num_nodes = len(adj_indptr) - 1

    # 创建稀疏邻接矩阵
    adj_row_indices = torch.repeat_interleave(torch.arange(num_nodes), adj_indptr[1:] - adj_indptr[:-1])
    adj_col_indices = adj_indices
    adj = torch.sparse_coo_tensor(
        torch.stack([adj_row_indices, adj_col_indices]),
        adj_data,
        size=(num_nodes, num_nodes)
    )
    a = adj.to_dense()
    adj_sparse = csr_matrix(a)
    adj = adj_sparse + adj_sparse.T.multiply(adj_sparse.T > adj_sparse) - adj_sparse.multiply(adj_sparse.T > adj_sparse)
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    a = a + a.t() - torch.diag(a.diagonal())
    a = a + torch.eye(a.size(0))
    deg_in = torch.pow(a.sum(0), -0.5)
    deg_mat_in = torch.diag(deg_in)
    a = a.matmul(deg_mat_in).matmul(deg_mat_in.T)

    # 数据集大小
    dataset_size = num_nodes

    # 随机打乱数据集索引
    indices = np.random.permutation(dataset_size)

    # 计算划分的索引位置
    train_end = int(dataset_size * 0.7)
    val_end = int(dataset_size * 0.9)

    # 划分数据集
    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    idx_train=torch.tensor(idx_train,dtype=torch.int64)
    idx_val=torch.tensor(idx_val,dtype=torch.int64)
    idx_test=torch.tensor(idx_test,dtype=torch.int64)

    # # 根据索引获取划分的数据集
    # train_set = dataset[train_indices]
    # val_set = dataset[val_indices]
    # test_set = dataset[test_indices]

    return adj, features, labels,idx_train,idx_val,idx_test,a