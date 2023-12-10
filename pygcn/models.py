import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


# nfeat=1433 底层节点的参数 feature的个数
# nhid=16 隐层节点个数
# nclass=7 最终的分类数
# dropout参数

class GCN(nn.Module):
    #GCN继承了Module这个类 Module为所有神经网络提供了一个模板
    def __init__(self, nfeat, nhid, nclass, dropout):
        # 这一行是一定要的，调用父类的一个初始化函数
        super(GCN, self).__init__()
        # def__init__（）函数是用于初始化的 用于传入参数后实例化一个模型
        # 真正训练和测试是调用forward函数
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout



    # x是输入特征，adj是邻接矩阵
    def forward(self, x, adj):
        # 继承nn.Module时forward子类一定要重写
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
        # 相当于x经过gc1,relu,dropout,gc2,log_softmax



