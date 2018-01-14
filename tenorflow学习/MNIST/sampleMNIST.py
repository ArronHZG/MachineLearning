# encoding: utf-8
'''
@author: arron
@license: (C) Copyright 2017-2025.
@contact: hou.zg@foxmail.com
@software: python
@file: sampleMNIST.py
@time: 2018/1/6 20:27
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
from tenorflow学习.MNIST.显示 import show
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
# 交叉熵  _表示一阶导数
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 使用反向传播算法(backpropagation algorithm)来有效地确定
# 你的变量是如何影响你想要最小化的那个成本值的
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
number = None
with tf.Session() as sess:
    sess.run(init)
    for i in range(1):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        show(batch_xs[50])
        # print(batch_xs)
        # print(batch_ys)
        # sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# fig, axs = plt.subplot(figsize=(5,5))
# axs.scatter(data)
# plt.show()



