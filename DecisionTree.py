"""
决策树代码实现 ID3

熵的概念：体系的混乱的程度
香农熵：一种信息的度量方式，表示信息的混乱程度，信息越有序，信息熵越低
信息增益：在划分数据集前后信息发生的变化称为信息增益
【寻找划分数据集的最好特征（划分之后信息熵最小，也就是信息增益最大的特征）】

"""
import numpy as np
import math
# 创建数据
def createDataset():
    # 两个特征【不浮出水面是否可以生存｜是否有脚蹼】
    # 标签 ：【 鱼类 ｜ 非鱼类 】 yes/no
    # labels 指的是 两个特征的名字 
    dataset = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfing','flippers']
    
    return dataset, labels

# 计算总的香农熵

def calShannonEnt(dataset):
    num = len(dataset)
    # 记录每种不同标签的样本数目
    labelCount = {}
    for feature in dataset:
        currentLabel = feature[-1]
        if currentLabel not in labelCount:
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    # 对于每个标签
    ShannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / num
        ShannonEnt -= prob * math.log(prob,2)
    return ShannonEnt

    

# 按照给定特征划分数据集,比如说查找“青年”数据集
def splitDataset(dataset,index,value):
    # index 标注不同的特征所在列
    # value 是需要返回的特征值
    # 返回的数据是index为某个value特征值的数据集【其中不包含index列】
    retDataset = []
    for feature in dataset:
        if feature[index] == value:
            reducedFeature = feature[:index]
            reducedFeature.extend(feature[index+1:])
            retDataset.append(reducedFeature)
    return retDataset

# 选择最好的数据集划分

def chooseBestFeatureToSplit(dataset):
    # 多少特征
    numFeature = len(dataset[0]) - 1
    print('numFeature: ',numFeature)
    # 原始香农熵
    baseEntropy = calShannonEnt(dataset)
    # 最优信息增益值和最优特征
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeature):
        # 获取所有数据中的当前feature值
        featureList = [example[i] for example in dataset]
        # 去重看有多少不同feature种类
        unique = set(featureList)
        # 以当前feature作为划分的信息增益
        newEntropy = 0.0
        for value in unique:
            # 不同的特征的数据集，比如青年数据集，老年数据集
            subDataset = splitDataset(dataset,i,value)
            # print(subDataset)
            prob = len(subDataset) / float(len(dataset))
            newEntropy += prob * calShannonEnt(subDataset)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
        print('infoGain=', infoGain, 'bestFeature=', bestFeature, baseEntropy, newEntropy)
    return bestFeature

# 选最多的label
def majorityCnt(classList):
    """majorityCnt(选择出现次数最多的一个结果)
    Args:
        classList label列的集合
    Returns:
        bestFeature 最优的特征列
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # print 'sortedClassCount:', sortedClassCount
    return sortedClassCount[0][0]

# 构建决策树

def createTree(dataset,labels):
    
    classList = [example[-1] for example in dataset]
    # 如果只有一个类别,直接返回就行
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 没有特征，直接返回label中最多的类别
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    # 否则选取最优的特征
    bestFeature = chooseBestFeatureToSplit(dataset)
    # print(labels,bestFeature)
    bestFeatureLabel = labels[bestFeature]

    # 初始化树
    myTree = {bestFeatureLabel:{}}
    del labels[bestFeature]
    
    featureValues = [example[bestFeature] for example in dataset]
    uniqueValues = set(featureValues)
    
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataset(dataset,bestFeature,value),subLabels)
    return myTree


# 测试分类

def classify(inputTree, featureLabels, testdata):
    # 第一个分类特征
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)
    key = testdata[featureIndex]
    value = secondDict[key]
    # print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', value)
    if isinstance(value,dict):
        # 还能再继续划分下去
        classLabel = classify(value,featureLabels,testdata)
    else:
        classLabel = value
    return classLabel
    



# 把决策树存起来

def storeTree(inputTree,filename):
    import pickle
    with open(filename,'wb') as f:
        pickle.dump(inputTree,f)

# 读取树

def loadTree(filename):
    import pickle
    f = open(filename,'rb')
    return pickle.load(f)

import copy
dataset,labels = createDataset()
testdata = [1,1]
myTree = createTree(dataset,copy.deepcopy(labels))
print(myTree)
label = classify(myTree,labels,testdata)
print(label)

