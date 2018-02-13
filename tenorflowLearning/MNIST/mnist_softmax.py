# encoding: utf-8
'''
@author: arron
@license: (C) Copyright 2017-2025.
@contact: hou.zg@foxmail.com
@software: python
@file: mnist_softmax.py
@time: 2018/1/6 20:27
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
from tenorflowLearning.MNIST.显示 import show

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 获取数据
# Create the model
x = tf.placeholder(tf.float32, [None, 784])  # 占位符
W = tf.Variable(tf.zeros([784, 10]))  # 权重 Variable变量 在这里称为张量
b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x,W) + b) #softmax后得到的分布律 this will be lead an error because of log(0)
y = tf.nn.log_softmax(tf.matmul(x, W) + b)
# Define loss and optimizer
# 交叉熵  y_表示实际分布
y_ = tf.placeholder("float", [None, 10])
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = -tf.reduce_sum(y_ * y)
# 使用反向传播算法(backpropagation algorithm)来有效地确定
# 你的变量是如何影响你想要最小化的那个成本值的
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 评估我们的模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
init = tf.initialize_all_variables()
#保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print(W.__dict__)
    save_path = saver.save(sess, "./model/model.ckpt")

# with tf.Session() as sess:
#     sess.run(init)
#     saver.restore(sess, "./model/model.ckpt")
#     with open("./张量.txt",'w+') as f:
#         f.write(str(W.eval()))
#     print("Model restored.")
#     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))