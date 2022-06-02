import matplotlib.pyplot as plt
import dataload
import numpy as np


def boundary(range, classifier, labels, values=[0]):
    x = np.arange(range[0], range[1], 0.1)
    y = np.arange(range[2], range[3], 0.1)
    xx, yy = np.meshgrid(x, y)
    xxx = xx.ravel()
    yyy = yy.ravel()
    zz = np.array(classifier(xx, yy))
    zz.reshape(x.shape)


def disp2D1(data, labels):
    for i in range(len(data)):
        if labels[i] == 1:
            plt.plot(data[i][0], data[i][1], "g+")
        else:
            plt.plot(data[i][0], data[i][1], "r*")
    plt.show()


def disp2D2(data1, data2, labels):
    # print(data1)
    # print(data2)
    # print(labels)
    for i in range(len(data1)):
        plt.plot(data1[i][0], data1[i][1], "r*")
    for i in range(len(data2)):
        plt.plot(data2[i][0], data2[i][1], "g+")
    plt.show()


if __name__ == "__main__":
    c1, c2, l = dataload.load2D("points_ring.pkl")
    disp2D2(c1, c2, l)

