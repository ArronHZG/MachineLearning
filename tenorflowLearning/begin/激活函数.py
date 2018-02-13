# -*- coding: utf-8 -*-
'''
@Author  : Arron
@email   :hou.zg@foxmail.com
@software: python
@File    : 激活函数.py
@Time    : 2018/2/12 20:07
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# def show(x_data,activation_function):
#     y_data=activation_function(x_data)
#     print(y_data)
#     # fig, ax = plt.subplots()
#     # ax.plot(y_data, label='test')
#     # ax.legend()
#     # import os
#     # plt.savefig(os.path.basename(__file__)[:-3] + ".png")
#     # plt.show()
#
# x_data = np.linspace(-1, 1, 500, dtype=np.float32)[:, np.newaxis]
# y_data=show(x_data,tf.nn.relu)
# init = tf.global_variables_initializer()  # 替换成这样就好
# with tf.Session as sess:
#     sess.run(init)
#     print(sess.run(tf.nn.relu(x_data)))

import numpy as np
import matplotlib.pyplot as plt
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

'''
tf.nn.relu()
tf.nn.sigmoid()
tf.nn.tanh()
tf.nn.elu()
tf.nn.bias_add()
tf.nn.crelu()
tf.nn.relu6()
tf.nn.softplus()
tf.nn.softsign()
tf.nn.dropout()
'''
x = np.arange(-10, 10, 0.01)

xt = tf.convert_to_tensor(x)
sess = tf.Session()
y = sess.run(tf.nn.dropout(xt))
plt.plot(x,y)
name="dropout"
plt.title(name+'()')
plt.savefig(name+'.png')
plt.show()