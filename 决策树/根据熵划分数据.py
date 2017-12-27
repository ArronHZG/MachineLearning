# encoding: utf-8
'''
@author: arron
@license: (C) Copyright 2017-2025.
@contact: hou.zg@foxmail.com
@software: python
@file: 根据熵划分数据.py
@time: 2017/12/27 8:06
'''
from math import log


def calcShannonEnt(dataSet):
    '''
    计算香农熵
    :param dataSet: 数据
    :return: 香农熵
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0
    # 计算每个信息在数据中出现的次数,获得概率
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        # print(prob)
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征值划分数据集
    :param dataSet: 数据集
    :param axis:数据内部特征下标
    :param value: 给定特征
    :return:划分后数据集
    '''
    retDataSet = []
    for featVec in dataSet:
        # print(featVec[axis])
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    根据香农熵选择最好的划分方式
    :param dataSet: 数据集
    :return:最好的划分方式 ,列表的索引号
    '''
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if infoGain > bestInfoGain:  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
        print("infoGain:{infoGain}".format(infoGain=infoGain))
    return bestFeature  # returns an integer



def createDataSet():
    dataSet = [[10, 10, 'yes'],
               [10, 10, 'yes'],
               [10, 5, 'no'],
               [5, 10, 'no'],
               [5, 10, 'no'],
               [5, 10, 'no'],
               [10, 10, 'maybe']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


if __name__ == '__main__':
    myData, labels = createDataSet()
    print(myData, labels)
    print(calcShannonEnt(myData))
    splitedTree=splitDataSet(myData,0,10)
    print("len:{}".format(len(splitedTree)),"ent:",calcShannonEnt(splitedTree))
    splitedTree = splitDataSet(myData, 0, 5)
    print("len:{}".format(len(splitedTree)),"ent:",calcShannonEnt(splitedTree))
    splitedTree = splitDataSet(myData, 1, 10)
    print("len:{}".format(len(splitedTree)),"ent:",calcShannonEnt(splitedTree))
    splitedTree = splitDataSet(myData, 1, 5)
    print("len:{}".format(len(splitedTree)),"ent:",calcShannonEnt(splitedTree))
    bestFeature=chooseBestFeatureToSplit(myData)
    print(bestFeature)