# -*- coding: utf-8 -*-
'''
@Author  : Arron
@email   :hou.zg@foxmail.com
@software: python
@File    : 优化器.py
@Time    : 2018/2/12 23:11

优化器
Optimizer基类提供了计算损失梯度和将梯度应用于变量的方法。一组子类实现了经典的优化算法，如GradientDescent和Adagrad。

你永远不会实例化Optimizer类本身，而是实例化其中一个子类。

tf.train.Optimizer
tf.train.GradientDescentOptimizer
tf.train.AdadeltaOptimizer
tf.train.AdagradOptimizer
tf.train.AdagradDAOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
tf.train.FtrlOptimizer
tf.train.ProximalGradientDescentOptimizer
tf.train.ProximalAdagradOptimizer
tf.train.RMSPropOptimizer


https://www.tensorflow.org/api_guides/python/train
'''