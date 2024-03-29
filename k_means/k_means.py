# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: k_means.py
@time: 2017/12/25 0025 23:13
'''

#################################################
# kmeans: k-means cluster
# Author : zouxy
# Date   : 2013-12-25
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import matplotlib.pyplot as plt


# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        print(index)
        centroids[i] = dataSet[index]
    print(centroids)
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    print(dataSet.shape)
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True
    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)
    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            ## for each centroid
            ## step 2: find the centroid who is closest
            # distance[x,for j in range(k):euclDistance(centroids[j], dataSet[i])]
            distance = [euclDistance(centroids[x], dataSet[i]) for x in range(k)]
            minDist=min(distance)
            minIndex=distance.index(minDist)
            for x in range(k):
                distance.append(euclDistance(centroids[x], dataSet[i]))

            # print(distance.index(max(distance)))
            # for j in range(k):
            #     distance[j] = euclDistance(centroids[j], dataSet[i])
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i] = minIndex,minDist
        ## step 4: update centroids
        for j in range(k):
            # print(clusterAssment[:, 0].A == j)
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j] = mean(pointsInCluster, axis=0)
    print('Congratulations, cluster complete!')
    return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()
