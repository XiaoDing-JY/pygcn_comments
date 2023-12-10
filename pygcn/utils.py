import numpy as np
import scipy.sparse as sp
import torch


# 将每篇论文的标签转换为one-hot编码形式
def encode_onehot(labels):
    # 提取每篇文章的一个标签 使之组成一个集合
    classes = set(labels)

    # np.identity生成一个单位矩阵
    # enumerate可以获得元素的索引和值
    # 这里是建立一个字典 key是文章类别 values是该类别所对应的一个独热码
    # 比如Reinforcement_Learning': array([1., 0., 0., 0., 0., 0., 0.]
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    # map是一个内置函数，它可以将一个函数应用到一个或多个可迭代对象的所有元素上，然后返回一个迭代器
    # 创建一个数组 对应着每篇文章的独独热码
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)

    # 最后返回的是每篇文章一个独热码的形式 labels_onehot.shape为(2708, 7)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora onl y for now)"""
    print('Loading {} dataset...'.format(dataset))

    # genfromtxt:从文本中加载数据并生成numpy数组 *args用于传入不定长参数
    # fname:路径   dtype:数据类型	str字符串 np.dtype(str)存储为文本
    # 先存储为str而不是float的原因应该是数据集的最后一列是文本
    # idx_features_labels.shapd: (2708, 1435)
    idx_features_labels = np.genfromtxt(fname="{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    # idx_features_labels[:, 1:-1] 左闭右开切片 取所有行 取1到-1列 也就是比原来少了两列
    # sp.csr_matrix用于存储稀疏矩阵
    # features.shape: (2708, 1433)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)


    # idx_features_labels[:, -1] 取最后一列 也就是每篇文章的类别出来 shape:2708
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 取出每篇论文的编号
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    # 因为数据集中的论文是有一个编号的 这里是用字典把论文的编号映射到0到2708
    idx_map = {j: i for i, j in enumerate(idx)}

    # 从文本文件中加载数据创建一个numpy数组 这里都是整数 所以dtype=np.int32
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    # edges_unordered.flatten()) 将原来(5429, 2)展开为转化为 (10858,)的一维数组
    # 将数据集中边的编号映射到idx_map中的编号
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # coo_matrix是 SciPy 库中用于表示稀疏矩阵的一种数据结构
    # sparse_matrix = coo_matrix((data, (row, col)), shape=(3, 5))
    # data表示非零元素的值 (row, col)表示非零元素所在的位置 shape=(3, 5)表示矩阵的大小
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix 建立邻接对称矩阵
    # 将非对称邻接矩阵转变为对称邻接矩阵（有向图转无向图）  可以推导一下公式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 对稀疏矩阵做一个行规范化
    features = normalize(features)

    # 对称邻接矩阵+单位矩阵，并进行归一化
    # 这里即是A波浪=A+I，添加了自连接的邻接矩阵
    adj = normalize(adj + sp.eye(adj.shape[0]))


    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # features.todense()是将稀疏矩阵转化为密集矩阵 再转化为tensor的形式
    features = torch.FloatTensor(np.array(features.todense()))

    # np.where(labels)返回非零元素的索引 我们取列的索引出来
    # one-hot向量label转常规label：0,1,2,3,
    labels = torch.LongTensor(np.where(labels)[1])


    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# 对一个稀疏矩阵进行行规范化 以确保每个节点（或行）的权重之和为1，从而使得矩阵更容易用于某些机器学习或图分析算法。
# 这里是计算D^-1A，而不是计算论文中的D^-1/2AD^-1/2
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# 只跑这个文件代码测试用
if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    print(adj)
    print(features)
    print(labels)
    print(idx_train)
    print(idx_val)
    print(idx_test)