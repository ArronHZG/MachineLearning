# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2018-2025, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: 保存图片.py
@time: 2018/1/24 0024 23:30
'''
import numpy as np
import struct
import matplotlib.pyplot as plt

filename = r'.\MNIST_data\t10k-images.idx3-ubyte'
binfile = open(filename, 'rb')
buf = binfile.read()
index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)  # 读取前4个字节的内容
index += struct.calcsize('>IIII')
im = struct.unpack_from('>784B', buf, index)  # 以大端方式读取一张图上28*28=784
index += struct.calcsize('>784B')
binfile.close()
im = np.array(im)
im = im.reshape(28, 28)
fig = plt.figure()
# plotwindow = fig.add_subplot(111)
plt.axis('off')
plt.imshow(im, cmap='gray')
plt.savefig("test.png")  # 保存成文件
plt.show()
plt.close()