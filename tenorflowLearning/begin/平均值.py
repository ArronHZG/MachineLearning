# -*- coding: utf-8 -*-
'''
@Author  : Arron
@email   :hou.zg@foxmail.com
@software: python
@File    : 平均值.py
@Time    : 2018/2/12 22:16
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(1, 10, 1)
xt = tf.constant([[1., 2.],
                  [3., 4.]])
sess = tf.Session()
# y = sess.run(tf.reduce_mean(xt))
print(sess.run(tf.reduce_mean(xt)))
print(sess.run(tf.reduce_mean(xt,reduction_indices=0)))
print(sess.run(tf.reduce_mean(xt,0)))
print(sess.run(tf.reduce_mean(xt,1)))
