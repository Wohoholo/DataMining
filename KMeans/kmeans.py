from math import sqrt
from random import randint
from matplotlib import pyplot as plt

def chooseCluster(k_classes, Data, centers):

    clusters = {k:[] for k in k_classes}

    for data in Data:
        dist = []

        for center in centers:
            dist.append(calcEuclideanDist(data, center))

        cluster = k_classes[dist.index(min(dist))]
        clusters[cluster].append(data)
    return clusters

def KMeans(k_class, Data, centers):
    # k = len(k_class)
    show(Data, centers)
    clusters = chooseCluster(k_class, Data, centers)
    plotClusters(clusters, centers)
    changed = True
    old_centers = centers
    while changed:
        new_centers = []
        for center in clusters.values():
            new_centers.append([sum(x[0] for x in center)/len(center), sum(y[1] for y in center)/len(center)])

        if all(a[0]==b[0] and a[1]==b[1] for a, b in zip(new_centers, old_centers)):
            print('END')
            break
        old_centers = new_centers
        clusters = chooseCluster(k_class, Data, new_centers)
        plotClusters(clusters, new_centers)

    plt.ioff()

    plt.show()
    return clusters


def calcEuclideanDist(vector_1, vector_2):

    return sqrt(sum(map(lambda x: (x[0]-x[1])**2, zip(vector_1, vector_2))))

def createData(k, num_data, dimension=2):
    data = []
    for i in range(num_data):
        tmp = []
        for j in range(dimension):
            tmp.append(randint(1, 100))
        data.append(tmp)

    centers = []

    for i in range(k):
        centers.append(data.pop(int(randint(0, num_data)/k)))

    return data, centers

def show(data, center):
    plt.scatter([x[0] for x in data], [y[1] for y in data])
    plt.pause(0.2)
    colors = ['r', 'g', 'y']
    for i in range(len(center)):
        x, y = center[i][0], center[i][1]
        color = colors[i]
        plt.scatter(x, y, c=color, marker='x')
        plt.pause(0.2)
    plt.show()

def plotClusters(cluster, center):
    colors = ['r', 'g', 'y']
    for key, value in cluster.items():
        for x, y in value:
            plt.scatter(x, y, c=key)
            plt.pause(0.2)
    for i in range(len(center)):
        x, y = center[i][0], center[i][1]
        color = colors[i]
        plt.scatter(x, y, c=color, marker='x')
        plt.pause(0.2)
    plt.show()

if __name__ == '__main__':
    fig = plt.figure()
    plt.ion()

    data, centers = createData(3, 100)

    KMeans(['r', 'g', 'y'], data, centers)
    plt.show()