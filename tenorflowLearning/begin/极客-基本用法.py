# encoding: utf-8
'''
@author: arron
@license: (C) Copyright 2017-2025.
@contact: hou.zg@foxmail.com
@software: python
@file: 极客-基本用法.py
@time: 2018/1/6 19:28
'''
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的 run() 方法初始化 'x'
x.initializer.run()

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
sub = tf.subtract(x, a)
print(sub.eval())

# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.global_variables_initializer()

# 启动图, 运行 op
with tf.Session() as sess:
    # 运行 'init' op
    sess.run(init_op)
    # 打印 'state' 的初始值
    print(sess.run(state))
    # 运行 op, 更新 'state', 并打印 'state'
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
# Fetch
# 为了取回操作的输出内容, 可以在使用
# Session 对象的 run() 调用 执行图时,
# 传入一些 tensor,
# 这些 tensor 会帮助你取回结果.
# 在之前的例子里, 我们只取回了单个节点 state,
# 但是你也可以取回多个 tensor:
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)# 3*(2+7)
with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)
# Feed

# 上述示例在计算图中引入了 tensor, 以常量或变量的形式存储.
#  TensorFlow 还提供了 feed 机制,
# 该机制 可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor.
#
# feed 使用一个 tensor 值临时替换一个操作的输出结果. 你可以提供 feed 数据作为 run() 调用的参数.
# feed 只在调用它的方法内有效, 方法结束, feed 就会消失. 最常见的用例是将某些特殊的操作指定为 "feed" 操作,
# 标记的方法是使用 tf.placeholder() 为这些操作创建占位符.


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
  print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))
  # for a larger - scale example of feeds.如果没有正确提供 feed, placeholder() 操作将会产生错误.MNIST 全连通 feed 教程 (source code) 给出了一个更大规模的使用 feed 的例子