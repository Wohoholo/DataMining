import math

class ID3:

    def __init__(self):
        pass

    def CreateDataSet(self):
        dataset = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return dataset, labels

    def calShannonEnt(self, dataset):
        numEntries = len(dataset)
        labelCounts = {}

        for featVec in dataset:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0

            labelCounts[currentLabel] += 1

        shannonEnt = 0.0

        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob*math.log(prob, 2)

        return shannonEnt

    def splitDataset(self, dataset, axis, value):
        retDataset = []
        for featVec in dataset:
            if featVec[axis] == value:
                #避免Python的特性 浅复制
                #去除特征本身
                reduceFeatVec = featVec[:axis]
                reduceFeatVec.extend(featVec[axis+1:])
                retDataset.append(reduceFeatVec)

        return retDataset


    def chooseBestFeatureToSplit(self, dataset):
        numberFeatures = len(dataset[0]) - 1
        baseEntropy = self.calShannonEnt(dataset)
        bestInfoGain = 0.0
        bestFeature = -1

        for i in range(numberFeatures):
            featList = [example[i] for example in dataset]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataset = self.splitDataset(dataset, i, value)
                prob = len(subDataset)/float(len(dataset))
                newEntropy += prob*self.calShannonEnt(subDataset)
            infoGain = baseEntropy - newEntropy

            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i

        return bestFeature

    def majorityCnt(self, classlist):
        classCount = {}
        for vote in classlist:
            if vote not in classCount:
                classCount[vote] = 0
            classCount[vote] += 1

        sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, dataset, labels):
        classlist = [example[-1] for example in dataset]
        if classlist.count(classlist[0]) == len(classlist):
            return classlist[0]

        if len(dataset[0]) == 1:
            return self.majorityCnt(classlist)

        bestFeat = self.chooseBestFeatureToSplit(dataset)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}

        del labels[bestFeat]
        featValues = [example[bestFeat] for example in dataset]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataset(dataset, bestFeat, value), subLabels)
        return myTree

if __name__ == '__main__':
    id3 = ID3()

    myData, labels = id3.CreateDataSet()
    print(id3.createTree(myData, labels))