# -*- coding: utf-8 -*-
'''
@Author  : Arron
@email   :hou.zg@foxmail.com
@software: python
@File    : FirstDNN.py
@Time    : 2018/2/12 14:50
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, name='layer', activation_function=None):
    '''
    增加神经网络层数
    :param inputs:输入数据
    :param in_size:输入神经元个数
    :param out_size:输出神经元个数
    :param activation_function:激活函数
    :return:返回数据
    '''
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='Weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='Biases')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 预测数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


# 神经网络结构 start
# 利用占位符定义我们所需的神经网络的输入。 tf.placeholder()就是代表占位符，
# 这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1。
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1],name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1],name='y_input')
# 搭建网络
# 使用 Tensorflow 自带的激励函数tf.nn.relu
l1 = add_layer(xs, 1, 10, name='hide',activation_function=tf.nn.relu)  # 隐藏层
# 接着，定义输出层。此时的输入就是隐藏层的输出——l1，输入有10层（隐藏层的输出层），输出有1层。
prediction = add_layer(l1, 10, 1, name='output  ', activation_function=None)  # 输出层
# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction,name='reduce_sum'), reduction_indices=[1]),name='reduce_mean')
# 接下来，是很关键的一步，如何让机器学习提升它的准确率。
# tf.train.GradientDescentOptimizer()中的值通常都小于1，这里取的是0.1，代表以0.1的效率来最小化误差loss。
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1,name='GradientDescentOptimizer').minimize(loss)
# 神经网络结构 end
# 定义Session，并用 Session 来执行 init 初始化步骤。 （注意：在tensorflow中，只有session.run()才会执行我们定义的运算。）
# 使用变量时，都要对它进行初始化，这是必不可少的。
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好
sess = tf.Session()
writer=tf.summary.FileWriter('./fistCNN_Logs')
writer.add_graph(sess.graph)
sess.run(init)
#绘图
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(x_data,y_data,color='b')
plt.ion()#动图
plt.show()
lines=ax.plot(x_data,y_data,'r-',lw=5)
plt.pause(0.1)
# 训练
# lines = []
for i in range(5000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # 每50步我们输出一下机器学习的误差。

    if i % 100 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        # ax.lines.remove(lines[0])
        prediction_value=sess.run(prediction,feed_dict={xs: x_data})
        lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)
# sess.close()
plt.pause(10)