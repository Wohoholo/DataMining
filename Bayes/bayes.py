from math import exp, sqrt, pi
from random import shuffle

def readData(filename):
    out_q = []
    with open(filename, 'r') as f:
        feat = f.readline().strip('\n').split(',')

        for line in f:
            line = line.strip('\n').split(',')
            line[1:] = [float(d) for d in line[1:]]
            out_q.append(line)
    # shuffle(out_q)
    return out_q, feat

def Guassian(in_q, x):
    if in_q.count(x) == len(in_q):
        return 1/len(in_q)
    mean = sum(in_q)/len(in_q)
    sgm = sqrt(sum(map(lambda x:(x-mean)**2, in_q))/len(in_q))
    # print(mean)
    # print(x)
    # print(sgm)
    X = (x-mean)/sgm
    prob = exp(-(X**2)/2) * 1/sqrt(2*pi)
    return prob

def processData(in_q, feat):
    Data = {}

    for line in in_q:
        # print(line)
        if line[0] not in Data:
            Data[line[0]] = {key:[] for key in feat}
            Data[line[0]]['num'] = 1

        else:
            Data[line[0]]['num'] += 1

        for key, value in zip(feat, line[1:]):
            Data[line[0]][key].append(value)

    return Data

def calBayes(Data, test, featVect):
    total = sum([Data[key]['num'] for key in Data])
    probs = [Data[key]['num']/total for key in Data]
    # print(Data)
    result = []
    classes = list(Data.keys())
    # print(classes)
    for j in range(len(classes)):
        prob = probs[j]
        cla = classes[j]
        for i in range(len(test)):
            # print(featVect)
            feat = featVect[i]
            # print(feat)
            prob *= Guassian(Data[cla][feat], test[i])

        result.append(prob)

    return result

if __name__ == '__main__':
    data, feat = readData('wine.data')
    shuffle(data)
    # print(data)
    test = []
    for i in range(10):
        line = data[i][:]
        line.pop(0)
        test.append(line)
    # print(test)
    dic = processData(data, feat)
    classes = list(dic.keys())
    class_result = []
    for t_data in test:
        print(t_data)
        result = calBayes(dic, t_data, feat)
        class_result.append(classes[result.index(max(result))])
    print(class_result)