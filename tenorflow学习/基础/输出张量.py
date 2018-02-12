# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2018-2025, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: 输出张量.py
@time: 2018/1/25 0025 19:46
'''
import tensorflow as tf

x = tf.Variable(tf.constant(0.1, shape=[10]))
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    with open("./张量.txt",'w+') as f:
        f.write(str(x.eval()))

# [0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]