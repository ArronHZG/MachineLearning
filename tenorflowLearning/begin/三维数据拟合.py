# encoding: utf-8
'''
@author: arron
@license: (C) Copyright 2017-2025.
@contact: hou.zg@foxmail.com
@software: python
@file: 三维数据拟合.py
@time: 2018/1/6 13:01
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))  # 随机输入
y_data = np.dot([1, 2], x_data) + 3

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
step = 0
while sess.run(loss) > 1e-12:
    sess.run(train)
    step += 1
print(step, sess.run(W), sess.run(b))


# # 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
# x, y, z = x_data[0], x_data[1], y_data
# ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
# #  将数据点分成三部分画，在颜色上有区分度
#
# # print(z)
#
# X, Y = np.meshgrid(x, y)
# Z = 1 * X + 2 * Y + 3
# # Plot the surface.
# surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# ax.scatter(x, y, z,s=50, c='b')
# # ax.set_zlabel('Z')  # 坐标轴
# # ax.set_ylabel('Y')
# # ax.set_xlabel('X')
# plt.show()  # # # with label
# plt.subplot(212)
# label = list(np.ones(20)) + list(2 * np.ones(15)) + list(3 * np.ones(15))
# label = np.array(label)
# plt.scatter(x[:, 1], x[:, 0], 15.0 * label, 15.0 * label)
#
# # with legend
# f2 = plt.figure(2)
# idx_1 = np.find(label == 1)
# p1 = plt.scatter(x[idx_1, 1], x[idx_1, 0], marker='x', color='m', label='1', s=30)
# idx_2 = np.find(label == 2)
# p2 = plt.scatter(x[idx_2, 1], x[idx_2, 0], marker='+', color='c', label='2', s=50)
# idx_3 = np.find(label == 3)
# p3 = plt.scatter(x[idx_3, 1], x[idx_3, 0], marker='o', color='r', label='3', s=15)
# plt.legend(loc='upper right')
