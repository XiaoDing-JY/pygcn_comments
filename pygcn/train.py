from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')


# Dropout是一种常用的神经网络正则化技术，它可以有效降低模型的过拟合风险。
# Dropout会随机地将神经网络中的部分神经元暂时“丢弃”（即将它们的输出置零）
# 从而减少神经元之间的依赖关系，促使网络学习到更加鲁棒和泛化能力更强的特征。
# 需要注意的是，当使用Dropout时，在模型进行推断（测试）时，一般不会应用Dropout
# 可以通过model.eval()来将模型切换到推断模式，此时Dropout将被自动关闭。
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#这里的np,torch,torch,cuda都是提前写好随机种子，后面生成的随机数都是一样的。
#设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
#使得每次运行该 .py 文件时生成的随机数相同。
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args.cuda)


# Load data 加载数据
# adj样本关系的对称邻接矩阵的稀疏张量
# features样本特征张量
# labels样本标签
# idx_train训练集索引列表
# idx_val验证集索引列表
# idx_test测试集索引列表
adj, features, labels, idx_train, idx_val, idx_test = load_data()



# Model and optimizer
# 这里只是传入一些参数 实例化一个对象 还没有前向传播
# nfeat=1433, int
# nhid=16, int, Number of hidden units,int
# nclass=7, int
# dropout=0.5, float, Dropout rate (1 - keep probability)
model = GCN(nfeat=features.shape[1], #1433 每篇文章所对应的特征向量长度
            nhid=args.hidden,
            nclass=labels.max().item() + 1,# item() 方法将这个张量的值转换为 Python 标量
            dropout=args.dropout)

# 可以打印模型的结构
print(model)
print(model.parameters())


# Adam算法结合了动量（Momentum）和自适应学习率（Adaptive Learning Rate）两个技术
# model.parameters()是模型的参数，让optimizer知道模型调整的参数有哪些
# ls是学习率 weight_decay是权重衰减
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


# 下面这两种方式都可以将数据转移到cuda上 主要是模型和数据 损失函数通常不用
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# if args.cuda:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(torch.cuda.get_device_name(0))
#     model = model.to(device)
#     features = features.to(device)
#     adj = adj.to(device)
#     labels = labels.to(device)
#     idx_train = idx_train.to(device)
#     idx_val = idx_val.to(device)
#     idx_test = idx_test.to(device)
#


def train(epoch):

    t = time.time()

    # 固定语句，主要针对启用BatchNormalization和Dropout
    model.train()

    # 将梯度信息清零
    optimizer.zero_grad()

    # 下面这行代码是前向传播
    # features和adj参数直接传到Module里面的forward函数
    # 前向传播后得到output
    output = model(features, adj)

    # 损失函数：需要将模型的输出与真实标签进行对比。对于每个样本，将其真实标签对应的概率取负对数，然后对所有样本求平均，得到最终的损失值。
    # 该损失值越小，表示模型的预测结果与真实标签越接近，模型的性能越好。
    # 在PyTorch中，通常情况下不需要显式地将损失函数移动到GPU上。
    # 当你在定义神经网络时，将整个模型移动到GPU上后，损失函数也会自动在相同的设备上进行计算。
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])

    # 反向传播 用于计算每个节点的一个参数
    loss_train.backward()

    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)


    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    # 固定模式 用于测试
    model.eval()
    #传入features, adj参数
    output = model(features, adj)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
print('complete')


# 优化器optimizer的example：
# for input, target in dataset:
#     optimizer.zero_grad() 将上一步的梯度进行清零 防止上一步对现在造成影响
#     output = model(input)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()
