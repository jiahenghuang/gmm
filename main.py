# -*- coding: utf-8 -*-
from gmm import *
import numpy as np

# 载入数据
with open('gmm.data') as fr:
    Y = fr.readlines()
Y = [[float(i.strip())] for i in Y]
Y = np.array(Y)
matY = np.matrix(Y, copy=True)

# 模型个数，即聚类的类别个数，有多少个高斯模型
K = 4
iters = 100  # em算法迭代次数

# 计算 GMM 模型参数
mu, cov, alpha = GMM_EM(matY, K, iters)

#打印mu
print(mu)

#打印协方差
print(cov)

#打印每个正态分布的分值
print(alpha)