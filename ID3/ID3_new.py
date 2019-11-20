from math import log
from random import randint


def calEntropy(dataset):
    Entropy = 0.0

    calDict = {}

    for line in dataset:
        if line[-1] not in calDict:
            calDict[line[-1]] = 0
        calDict[line[-1]] += 1

    for key in calDict.keys():
        prob = float(calDict[key]/len(dataset))
        Entropy -= prob * log(prob, 2)

    return Entropy

def splitData(dataset, axis, value):
    reduceData = []

    for line in dataset:
        if line[axis] == value:
            tmp = line[:axis]
            tmp.extend(line[axis+1:])

            reduceData.append(tmp)
    return reduceData


def chooseBestFeature(dataset):
    baseEntropy = calEntropy(dataset)
    baseGain = 0.0
    bestFeat = -1

    featNum = len(dataset[0]) - 1

    for i in range(featNum):
        uniqueVal = []
        Entropy = 0.0

        for line in dataset:
            uniqueVal.append(line[i])
        uniqueVal = set(uniqueVal)

        for val in uniqueVal:

            data = splitData(dataset, i, val)
            prob = len(data)/len(dataset)

            Entropy += prob * calEntropy(data)

        gain = baseEntropy - Entropy
        if gain >= baseGain:
            baseGain = gain
            bestFeat = i

    return bestFeat

def majority(classlist):
    classCount = {}

    for vote in classlist:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClass = sorted(classCount.items(), key=lambda item:item[0], reverse=True)
    return sortedClass[0][0]


def createTree(dataset, labels):
    # print(dataset)
    classlist = [example[-1] for example in dataset]

    if classlist.count(classlist[0]) == len(dataset):
        return classlist[0]

    if len(dataset[0]) == 1:
        return majority(classlist)



    bestFeat = chooseBestFeature(dataset)
    bestFeatVal = labels[bestFeat]
    myTree = {bestFeatVal:{}}

    del labels[bestFeat]
    featVals = [example[bestFeat] for example in dataset]
    uniqueVal = set(featVals)

    for featVal in uniqueVal:
        subLabels = labels[:]

        myTree[bestFeatVal][featVal] = createTree(splitData(dataset, bestFeat, featVal), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):

    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = -1

    for key in secondDict.keys():

        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel

def classifyAll(inputTree, featLabels, testDataset):
    classLabel = []
    for data in testDataset:
        classLabel.append(classify(inputTree, featLabels, data))

    return classLabel

def createDataSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    dataSet = [[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels

def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    testSet = [[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]]
    return testSet

def createIrisData():
    dataset = []

    with open('bezdekIris.data', 'r') as f:
        labels = f.readline().strip('\n').split(',')

        for line in f:
            dataset.append(line.strip('\n').split(','))


    return dataset, labels

def testDataset():
    dataset = []
    n = 5

    with open('bezdekIris.data', 'r') as f:

        a = f.readlines()

        for i in range(n):
            dataset.append(a[randint(1, 150)].strip('\n').split(',')[:-1])

    return dataset

if __name__ == '__main__':

    dataset, labels = createIrisData()
    myTree = createTree(dataset, labels[:])
    print(myTree)
    testSet = testDataset()
    print(testSet)
    output = classifyAll(myTree, labels, testSet)
    print(output)
