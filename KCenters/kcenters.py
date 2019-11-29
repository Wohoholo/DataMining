from random import randint
from matplotlib import pyplot as plt

def calManhattanDistance(vector_1, vector_2):
    return sum([abs(x1-x2) for x1, x2 in zip(vector_1, vector_2)])

def chooseCluster(k_classes, Data, centers):
    clusters = {k:[] for k in k_classes}
    cost = 0
    for data in Data:
        dist = []

        for center in centers:
            dist.append(calManhattanDistance(data, center))

        cost += min(dist)
        cluster = k_classes[dist.index(min(dist))]
        clusters[cluster].append(data)

    return clusters, cost

def chooseCenters(centers, k, Data, l, r, list):


    if len(list) == k:
        centers.append(list[:])
        return
    for i in range(l, r):
        temp = list[:]
        temp.append(Data[i])
        chooseCenters(centers, k, Data, i+1, r, temp)


def KCenters(k_classes, Data, centers):

    bestCenter = None
    minCost = 9999
    show(data)

    for center in centers:

        cluster, cost = chooseCluster(k_classes, Data, center)

        if cost < minCost:
            minCost = cost
            bestCenter = center

            plotClusters(k_classes, cluster, center)
        # print(cost, cluster)

    print('out!')
    plt.ioff()
    # plt.show()

    return bestCenter, cluster

def createData(k, num_data, dimension=2):

    data = []
    for i in range(num_data):
        tmp = []
        for j in range(dimension):
            tmp.append(randint(1, 100))
        data.append(tmp)

    centers = []

    chooseCenters(centers, k, data, 0, num_data, [])

    return data, centers

def show(Data):


    for x, y in Data:
        plt.scatter(x, y, c='b')

    plt.pause(1)
    plt.show()

def plotClusters(colors, cluster, center):

    # plt.clf()
    # for i in range(len(center)):
    #     x, y = center[i][0], center[i][1]
    #     color = colors[i]
    #     plt.scatter(x, y, c=color, marker='x')
    #     plt.pause(0.1)
    plt.clf()
    for key, value in cluster.items():
        x_aix = list(map(lambda pos:pos[0], value))
        y_aix = list(map(lambda pos:pos[1], value))
        plt.scatter(x_aix, y_aix, c=key)
    plt.pause(1)
    plt.show()


if __name__ == '__main__':


    classes = ['g', 'b', 'r']
    data, centers = createData(len(classes), 100)
    # print(centers)
    plt.ion()
    fig = plt.figure(1)
    center, cluster = KCenters(classes, data, centers)
    plt.show()
