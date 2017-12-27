# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: test_kmeans.py
@time: 2017/12/25 0025 23:19
'''
#################################################
# kmeans: k-means cluster
# Author : zouxy
# Date   : 2013-12-25
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import math
from k_means import kmeans,showCluster
import time
import matplotlib.pyplot as plt

## step 1: load data
print("step 1: load data...")
dataSet = []
fileIn = open('./testSet.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    # dataSet.append((int(math.floor(float(lineArr[0]))), int(math.floor(float(lineArr[1])))))
    dataSet.append((float(lineArr[0]), float(lineArr[1])))
# print(dataSet)
## step 2: clustering...
print("step 2: clustering...")
dataSet = mat(dataSet)
# print(dataSet)
k = 4
centroids, clusterAssment = kmeans(dataSet, k)

## step 3: show the result
print("step 3: show the result...")
showCluster(dataSet, k, centroids, clusterAssment)