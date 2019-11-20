from math import log
from random import randint

class C45:

    def __init__(self):
        pass

    def calShannonEnt(self, dataset):

        numEntries = len(dataset)

        labelCounts = {}

        for feature in dataset:
            featLabel = feature[-1]

            if featLabel not in labelCounts:
                labelCounts[featLabel] = 0

            labelCounts[featLabel] += 1

        shannonEnt = 0.0

        for key in labelCounts.keys():
            prob = float(labelCounts[key]/numEntries)

            shannonEnt -= prob * log(prob, 2)

        return shannonEnt


    def splitDataset(self, dataset, axis, value):
        retDataset = []

        for feature in dataset:
            if feature[axis] == value:
                reduceFeature = feature[:axis]
                reduceFeature.extend(feature[axis+1:])
                retDataset.append(reduceFeature)

        return retDataset

    def chooseBestFeatureToSplit(self, dataset):

        numFeatures = len(dataset[0])-1
        baseEntropy = self.calShannonEnt(dataset)
        baseInfoGainRatio = 0.0
        bestFeature = -1

        for i in range(numFeatures):
            featList = [example[i] for example in dataset]
            uniqueVals = set(featList)
            newEntropy = 0.0
            splitInfo = 0.0

            for value in uniqueVals:
                subDataset = self.splitDataset(dataset, i, value)
                prob = float(len(subDataset))/float(len(dataset))
                newEntropy += prob * self.calShannonEnt(subDataset)
                splitInfo -= prob * log(prob, 2)

            infoGain = baseEntropy - newEntropy

            if splitInfo == 0: #fix the overflow bug
                continue

            infoGainRatio = infoGain / splitInfo
            if infoGainRatio > baseInfoGainRatio:
                baseInfoGainRatio = infoGainRatio
                bestFeature = i

        return bestFeature

    def majorityCnt(self, classList):

        classCount = {}

        for vote in classList:
            if vote not in classCount:
                classCount[vote] = 0

            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=lambda item:item[1], reverse=True)
        return sortedClassCount

    def createTree(self, dataset, labels):

        classList = [example[-1] for example in dataset]

        if classList.count(classList[0]) == len(classList):
            return classList[0]

        if len(dataset) == 1:
            return self.majorityCnt(classList)


        bestFeat = self.chooseBestFeatureToSplit(dataset)
        bestFeatLabel = labels[bestFeat]

        myTree = {bestFeatLabel:{}}

        del labels[bestFeat]

        featVals = [example[bestFeat] for example in dataset]
        uniqueVals = set(featVals)

        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataset(dataset, bestFeat, value), subLabels)

        return myTree

    def classify(self, inputTree, featLabels, testVec):
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        classLabel = -1

        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]

        return classLabel

    def classifyAll(self, inputTree, featLabels, testDataSet):
        """
        输入：决策树，分类标签，测试数据集
        输出：决策结果
        描述：跑决策树
        """
        classLabelAll = []
        for testVec in testDataSet:
            classLabelAll.append(self.classify(inputTree, featLabels, testVec))
        return classLabelAll

    def storeTree(self, inputTree, filename):
        """
        输入：决策树，保存文件路径
        输出：
        描述：保存决策树到文件
        """
        import pickle
        fw = open(filename, 'wb')
        pickle.dump(inputTree, fw)
        fw.close()

    def grabTree(self, filename):
        """
        输入：文件路径名
        输出：决策树
        描述：从文件读取决策树
        """
        import pickle
        fr = open(filename, 'rb')
        return pickle.load(fr)

    def createDataSet(self):
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

    def createTestSet(self):
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
    c = C45()
    dataset, labels = createIrisData()
    labels_tmp = labels[:]
    decisionTree = c.createTree(dataset, labels_tmp)
    print(decisionTree)
    testDataset = testDataset()
    print(testDataset)
    print(c.classifyAll(decisionTree, labels, testDataset))