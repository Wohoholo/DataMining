import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def createData(num_data=20):
    for i in range(num_data):
        x1 = round(random.random(), 2)
        x2 = round(random.random(), 2)
        y = (x1-1)**4 + 2*(x2**2)
        yield [x1, x2, y]



if __name__ == '__main__':

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.ion()
    # x1 = np.arange(-1, 1, 0.1)
    # x2 = np.arange(-1, 1, 0.1)
    # x1, x2 = np.meshgrid(x1, x2)
    # y = np.power((x1-1), 4) + 2*np.power(x2, 2)
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # # ax.plot_surface(x1, x2, y, rstride=1,cstride=1,cmap='rainbow')
    # # plt.show()
    #
    data = createData()
    f = open('data.txt', 'w')
    for x, y, z in data:
        f.write('{}  {}  {}\n'.format(x, y, z))
