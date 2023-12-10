import math
import torch

from torch.nn.parameter import Parameter
# 上下两句的import是等价的
from torch.nn.modules.module import Module

from torch.nn import  Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        ####################################参数的定义####################################
        # 先转化为张量，再转化为可训练的Parameter对象
        # Parameter用于将参数自动加入到参数列表
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 参数随机初始化函数
        # size包括(in_features, out_features)，size(1)应该是指out_features
        stdv = 1. / math.sqrt(self.weight.size(1))
        # weight在区间(-stdv, stdv)之间均匀分布随机初始化
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # torch.nn是矩阵相乘
        support = torch.mm(input, self.weight)
        # torch.spmm是稀疏矩阵乘法，说白了还是乘法而已，只是减小了运算复杂度
        output = torch.spmm(adj, support)
        # 有偏置的话加一个偏置
        if self.bias is not None:
            return output + self.bias
        else:
            return output


    # 当调用print函数时 会触发该方法
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
