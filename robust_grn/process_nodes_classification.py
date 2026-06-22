import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Actor
import torch
from torch_geometric.utils import to_networkx
import os.path as osp
from robust_grn.utils import *
def process_Actor():
    dataset = Actor(root='./dataset')

    # 获取第一个图数据
    data = dataset[0]

    # 获取掩码
    test_mask = data.test_mask  ###test_mask.sum(0)+train_mask.sum(0)+val_mask.sum(0)=x.shape(0)
    train_mask = data.train_mask
    val_mask = data.val_mask
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    test_mask_first_split = test_mask[:, 0]
    idx_test = torch.nonzero(test_mask_first_split).squeeze()
    train_mask_first_split = train_mask[:, 0]
    idx_train = torch.nonzero(train_mask_first_split).squeeze()
    val_mask_first_split = val_mask[:, 0]
    idx_val = torch.nonzero(val_mask_first_split).squeeze()

    features = data.x.to(device)
    labels = data.y.to(device)
    G = to_networkx(data, to_undirected=True)

    # 获得无向图的边索引
    edge_index = data.edge_index

    # 获取节点数量
    num_nodes = data.num_nodes

    values = torch.ones_like(edge_index[0], dtype=torch.float32)  # 边权重设为1，可根据需求修改

    # 构建稀疏张量的形状
    shape = (num_nodes, num_nodes)
    # 构建稀疏张量的邻接矩阵
    adj = torch.sparse_coo_tensor(edge_index, values, shape)
    a = adj.to_dense()

    a = a + a.t() - torch.diag(a.diagonal())
    a = a + torch.eye(a.size(0))
    deg_in = torch.pow(a.sum(0), -0.5)
    deg_mat_in = torch.diag(deg_in)
    a = a.matmul(deg_mat_in).matmul(deg_mat_in.T)
    # 稀疏矩阵处理
    # 将 edge_index 转换为稀疏邻接矩阵的行和列索引
    row = edge_index[0]
    col = edge_index[1]

    # 创建稀疏邻接矩阵
    adj_sparse = csr_matrix((np.ones(row.shape), (row, col)), shape=(num_nodes, num_nodes))
    adj = adj_sparse + adj_sparse.T.multiply(adj_sparse.T > adj_sparse) - adj_sparse.multiply(adj_sparse.T > adj_sparse)
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels,idx_train,idx_val,idx_test,a
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
    dataset = load_npz('dataset/other/amazon_electronics_photo.npz')
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
def process_coauthor_cs():
    dataset = load_npz('dataset/other/coauthor_cs.npz')
    features = dataset['node_attr']
    features_coo = features.tocoo()

    features = torch.sparse_coo_tensor(
        torch.LongTensor([features_coo.row, features_coo.col]),
        torch.FloatTensor(features_coo.data),
        torch.Size(features_coo.shape)
    )

    features = features.to_dense()

    adj = dataset['adj_matrix']
    adj_coo = adj.tocoo()

    adj_coo = torch.sparse_coo_tensor(
        torch.LongTensor([adj_coo.row, adj_coo.col]),
        torch.FloatTensor(adj_coo.data),
        torch.Size(adj_coo.shape))
    a = adj_coo.to_dense()
    labels = dataset['node_label']
    labels = torch.tensor(labels, dtype=torch.int64)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    a = a + a.t() - torch.diag(a.diagonal())
    a = a + torch.eye(a.size(0))
    deg_in = torch.pow(a.sum(0), -0.5)
    deg_mat_in = torch.diag(deg_in)
    a = a.matmul(deg_mat_in).matmul(deg_mat_in.T)

    # 数据集大小
    dataset_size = features.shape[0]

    # 随机打乱数据集索引
    indices = np.random.permutation(dataset_size)

    # 计算划分的索引位置
    train_end = int(dataset_size * 0.7)
    val_end = int(dataset_size * 0.9)

    # 划分数据集
    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    idx_train = torch.tensor(idx_train, dtype=torch.int64)
    idx_val = torch.tensor(idx_val, dtype=torch.int64)
    idx_test = torch.tensor(idx_test, dtype=torch.int64)
    return adj, features, labels,idx_train,idx_val,idx_test,a
def process_blogcatalog():
    dataset=load_npz('dataset/other/blogcatalog.npz')
    features = dataset['node_attr']
    features_coo = features.tocoo()

    features = torch.sparse_coo_tensor(
        torch.LongTensor([features_coo.row, features_coo.col]),
        torch.FloatTensor(features_coo.data),
        torch.Size(features_coo.shape)
    )

    features = features.to_dense()

    adj = dataset['adj_matrix']
    adj_coo = adj.tocoo()

    adj_coo = torch.sparse_coo_tensor(
        torch.LongTensor([adj_coo.row, adj_coo.col]),
        torch.FloatTensor(adj_coo.data),
        torch.Size(adj_coo.shape))
    a = adj_coo.to_dense()
    labels = dataset['node_label']
    labels = torch.tensor(labels, dtype=torch.int64)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    a = a + a.t() - torch.diag(a.diagonal())
    a = a + torch.eye(a.size(0))
    deg_in = torch.pow(a.sum(0), -0.5)
    deg_mat_in = torch.diag(deg_in)
    a = a.matmul(deg_mat_in).matmul(deg_mat_in.T)

    # 数据集大小
    dataset_size = features.shape[0]

    # 随机打乱数据集索引
    indices = np.random.permutation(dataset_size)

    # 计算划分的索引位置
    train_end = int(dataset_size * 0.7)
    val_end = int(dataset_size * 0.9)

    # 划分数据集
    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    idx_train = torch.tensor(idx_train, dtype=torch.int64)
    idx_val = torch.tensor(idx_val, dtype=torch.int64)
    idx_test = torch.tensor(idx_test, dtype=torch.int64)
    return adj, features, labels,idx_train,idx_val,idx_test,a
def process_Amazon_CS():
    dataset=load_npz('dataset/other/amazon_cs.npz')
    features = dataset['node_attr']
    features_coo = features.tocoo()

    features = torch.sparse_coo_tensor(
        torch.LongTensor([features_coo.row, features_coo.col]),
        torch.FloatTensor(features_coo.data),
        torch.Size(features_coo.shape)
    )

    features = features.to_dense()

    adj = dataset['adj_matrix']
    adj_coo = adj.tocoo()

    adj_coo = torch.sparse_coo_tensor(
        torch.LongTensor([adj_coo.row, adj_coo.col]),
        torch.FloatTensor(adj_coo.data),
        torch.Size(adj_coo.shape))
    a = adj_coo.to_dense()
    labels = dataset['node_label']
    labels = torch.tensor(labels,dtype=torch.int64)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    a = a + a.t() - torch.diag(a.diagonal())
    a = a + torch.eye(a.size(0))
    deg_in = torch.pow(a.sum(0), -0.5)
    deg_mat_in = torch.diag(deg_in)
    a = a.matmul(deg_mat_in).matmul(deg_mat_in.T)

    # 数据集大小
    dataset_size = features.shape[0]

    # 随机打乱数据集索引
    indices = np.random.permutation(dataset_size)

    # 计算划分的索引位置
    train_end = int(dataset_size * 0.7)
    val_end = int(dataset_size * 0.9)

    # 划分数据集
    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    idx_train = torch.tensor(idx_train, dtype=torch.int64)
    idx_val = torch.tensor(idx_val, dtype=torch.int64)
    idx_test = torch.tensor(idx_test, dtype=torch.int64)
    return adj, features, labels,idx_train,idx_val,idx_test,a


