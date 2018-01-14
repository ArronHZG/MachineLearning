# encoding: utf-8
'''
@author: arron
@license: (C) Copyright 2017-2025.
@contact: hou.zg@foxmail.com
@software: python
@file: 显示.py
@time: 2018/1/6 21:04
'''
import numpy as np
import matplotlib.pyplot as plt

def show(num):
    num = np.array(num).reshape((28, 28))
    data = []
    for i in range(28):
        for j in range(28):
            # print(i, j)
            if num[i][j] != 0:
                data.append((i, j, num[i][j]))

    data = np.array(data)
    # print(data)
    plt.scatter(data[:, 0], data[:, 1])
    plt.xticks(())
    plt.yticks(())
    plt.show()
# n=500
# x = np.random.normal(0, 1, n)
# y = np.random.normal(0, 1, n)
#
# # 计算颜色值
# color = np.arctan2(y, x)
# print(color)
# # 绘制散点图
# plt.scatter(x, y, s = 75, c = color, alpha = 0.5)
# # 设置坐标轴范围
# plt.xlim((-1.5, 1.5))
# plt.ylim((-1.5, 1.5))
#
# # 不显示坐标轴的值
# plt.xticks(())
# plt.yticks(())
#
# plt.show()
