# -*- coding: utf-8 -*-
'''
@Author  : Arron
@email   :hou.zg@foxmail.com
@software: python
@File    : mnist_CNN_summaries.py
@Time    : 2018/2/25 22:24
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class FLAGS:
    file_name = __file__[__file__.rfind('/') + 1:-3]
    log_dir = './logs/' + file_name
    keep_prob = 1.0


class CNN:

    def __init__(self):

        # ---------------定义函数----------------- #

        def weight_variable(shape):
            """
            权重初始化函数
            :return: 输出服从截尾正态分布的随机值
            """
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            """
            偏置初始化函数
            :return:
            """
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            """
            创建卷积op
            x 是一个4维张量，shape为[train_batch,height,width,channels]
            卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点
            :param W:
            :return:
            """
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

        def max_pool_2x2(x):
            """
            创建池化op
            采用最大池化，也就是取窗口中的最大值作为结果
            x 是一个4维张量，shape为[train_batch,height,width,channels]
            ksize表示pool窗口大小为2x2,也就是高2，宽2
            strides，表示在height和width维度上的步长都为2
            """
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

        def cnn_layer(input_tensor, patch_size, input_dim, output_dim, layer_name, act=tf.nn.relu):
            """Reusable code for making a simple neural net layer.

            It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([patch_size, patch_size, input_dim, output_dim])
                    variable_summaries(weights)
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
                    variable_summaries(biases)
                # 卷积神经网络
                with tf.name_scope('conv'):
                    conv = act(conv2d(input_tensor, weights) + biases)
                    tf.summary.histogram('conv', conv)
                return conv

        def max_pooling_layer(input_tensor, layer_name):
            with tf.name_scope(layer_name):
                pool = max_pool_2x2(input_tensor)
                tf.summary.histogram('pool', pool)
                return pool

        def fully_connected_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim])
                    variable_summaries(weights)
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
                    variable_summaries(biases)
                with tf.name_scope('max_pooling_flat'):
                    pool_flat = tf.reshape(input_tensor, [-1, input_dim])
                    activations = act(tf.matmul(pool_flat, weights) + biases)
                return activations

        def dropout_layer(input_tensor, keep_prob, layer_name):
            # with tf.name_scope(layer_name):

            dropped = tf.nn.dropout(input_tensor, keep_prob)
            return dropped

        def output_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.softmax):
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim])
                    variable_summaries(weights)
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
                    variable_summaries(biases)
                with tf.name_scope('softmax'):
                    output = act(tf.matmul(input_tensor, weights) + biases)
                return output

        def train_layer(tensor_true, tensor_pro, layer_name):
            # 预测值和真实值之间的交叉墒
            with tf.name_scope(layer_name):
                cross_entropy = -tf.reduce_sum(tensor_true * tf.log(tensor_pro), name='loss')
                tf.summary.scalar('cross_entropy', cross_entropy)
                # train op, 使用ADAM优化器来做梯度下降。学习率为0.0001
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
                return train_step

        def evaluate_layer(tensor_true, tensor_pro, layer_name):
            # 评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。
            # 因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
            with tf.name_scope(layer_name):
                correct_predict = tf.equal(tf.argmax(tensor_pro, 1), tf.argmax(tensor_true, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"), name='accuracy')
                tf.summary.scalar('accuracy', accuracy)
                return accuracy

        # ---------------获取数据----------------- #
        mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

        # ---------------第一层：输入层----------------- #
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

        with tf.name_scope('x_image'):
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            tf.summary.image('x_image', x_image,100)

        # ---------------第二层：卷积层+池化层----------------- #
        conv1 = cnn_layer(x_image, 5, 1, 32, 'conv_1', tf.nn.relu)
        pool1 = max_pooling_layer(conv1, 'pooling_1')
        # ---------------第三层：卷积层+池化层----------------- #
        conv2 = cnn_layer(pool1, 5, 32, 64, 'conv2', tf.nn.relu)
        pool2 = max_pooling_layer(conv2, 'pooling_2')
        # ---------------第四层：全连接层+dropout层----------------- #
        fully_connected = fully_connected_layer(pool2, 7 * 7 * 64, 1024, 'fully_connection', tf.nn.relu)
        # with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)
        dropout = dropout_layer(fully_connected, keep_prob, 'dropout')

        # ---------------第五层：输出层----------------- #
        output = output_layer(dropout, 1024, 10, 'output', tf.nn.softmax)

        # ---------------第六层：训练层----------------- #
        train_step = train_layer(y_, output, 'train')

        # ---------------第七层：评估层----------------- #
        accuracy = evaluate_layer(y_, output, 'evaluate')
        # ---------------开始训练----------------- #
        print("--------------------------train start--------------------------")
        with tf.Session() as sess:
            # ---------------绘制tensorboard----------------- #
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
            sess.run(tf.global_variables_initializer())
            for i in range(2001):
                train_batch = mnist.train.next_batch(100)
                train_step.run(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})
                test_batch = mnist.test.next_batch(50)
                if i % 10 == 0:
                    # 每100次输出一次日志
                    summary, train_accuracy = sess.run([merged, accuracy],
                                                       feed_dict={x: train_batch[0], y_: train_batch[1],
                                                                  keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                    train_writer.add_summary(summary, i)

                    summary, train_accuracy = sess.run([merged, accuracy],
                                                       feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
                    print("step %d, test accuracy %g" % (i, train_accuracy))
                    test_writer.add_summary(summary, i)

            print("test accuracy %g" % accuracy.eval(
                feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
            train_writer.close()
            test_writer.close()
        print("--------------------------train end--------------------------")


if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    CNN()
