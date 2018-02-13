# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2018-2025, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: 常量.py
@time: 2018/2/11 0011 23:29
'''
import tensorflow as tf

# create two matrixes

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)
# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)