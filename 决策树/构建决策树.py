# encoding: utf-8
'''
@author: arron
@license: (C) Copyright 2017-2025.
@contact: hou.zg@foxmail.com
@software: python
@file: 构建决策树.py
@time: 2017/12/27 17:24
'''

from 根据熵划分数据 import createDataSet,chooseBestFeatureToSplit,splitDataSet
def majorityCnt(classList):
    '''
    给定列表,返回众数
    :param classList: classList
    :return: 众数
    '''
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return max(classCount.items(), key=lambda x: x[1])[0]


def createTree(dataSet, labels):
    '''

    :param dataSet:
    :param labels:
    :return:
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def storeMyTree():
    myData, labels = createDataSet()
    print(myData, labels)
    myTree = createTree(myData, labels)
    print(myTree)
    return myTree

if __name__ == '__main__':
    # a="12345678901234561111"
    # a=list(a)
    # print(majorityCnt(a))
    myData, labels = createDataSet()
    print(myData, labels)
    myTree=createTree(myData, labels)
    print(myTree)