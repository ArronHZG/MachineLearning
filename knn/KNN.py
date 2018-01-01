# -*- coding: UTF-8 -*-
'''
@author: Arron
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: hou.zg@foxmail.com
@software: import
@file: KNN.py
@time: 2017/12/26 0026 1:24
'''

from numpy import *


# create a dataset which contains 4 samples with 2 classes
def createDataSet():
    # create a matrix: each row as a sample
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']  # four samples and two classes
    return group, labels


# classify using kNN
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    # step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet

    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    # squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(diff ** 2, axis=1)  # sum is performed by row
    distance = sum(diff ** 2, axis=1) ** 0.5

    # step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        # step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        # step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # step 5: the max voted class will return
    # print(classCount)
    # print(max(classCount.items(), key=lambda x: x[1])[0])
    # print([classCount.items(), lambda x: x[1]])
    # print(classCount.items())
    return max(classCount.items(), key=lambda x: x[1])[0]
    # maxCount = 0
    # for key, value in classCount.items():
    #     if value > maxCount:
    #         maxCount = value
    #         maxIndex = key
    # return maxIndex


if __name__ == '__main__':
    dataSet, labels = createDataSet()

    testX = array([1.2, 1.0])
    k = 3
    outputLabel = kNNClassify(testX, dataSet, labels, 3)
    print("Your input is:", testX, "and classified to class: ", outputLabel)

    testX = array([0.1, 0.3])
    outputLabel = kNNClassify(testX, dataSet, labels, 3)
    print("Your input is:", testX, "and classified to class: ", outputLabel)
