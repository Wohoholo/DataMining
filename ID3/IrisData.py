
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


        f.readline()

        for line in f:
            dataset.append(line.strip('\n').split(',')[:-1])

    return dataset
