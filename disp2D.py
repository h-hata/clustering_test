import matplotlib.pyplot as plt
import dataload
import numpy as np


def disp3D(xx, yy, zz):
    fig = plt.figure(figsize=(8, 8))  # unit inch

    # contour
    ax1 = fig.add_subplot(221, facecolor="w")
    ax1.set_aspect("equal")
    ax1.set_title("contour")
    ax1.contour(xx, yy, zz, np.arange(0.01, 0.14, 0.01))

    # surface
    ax2 = fig.add_subplot(222, projection="3d", facecolor="w")
    ax2.set_title("surface")
    # ax2.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax2.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap="plasma")
    # ax2.plot_surface(xx, yy, zz, rstride=1, cstride=1, color="green")

    # fireframe
    ax3 = fig.add_subplot(223, projection="3d", facecolor="w")
    ax3.set_title("wireframe")
    ax3.plot_wireframe(
        xx, yy, zz, linewidth=0.5, color="green",
    )

    plt.show()


def boundary(range, classifier, labels, values=[0]):
    x = np.arange(range[0], range[1], 0.1)
    y = np.arange(range[2], range[3], 0.1)
    xx, yy = np.meshgrid(x, y)
    # xxx = xx.ravel()
    # yyy = yy.ravel()
    zz = np.array(classifier(xx, yy))
    zz.reshape(x.shape)


def disp2D1(data, labels):
    for i in range(len(data)):
        if labels[i] == 1:
            plt.plot(data[i][0], data[i][1], "g+")
        else:
            plt.plot(data[i][0], data[i][1], "r*")
    plt.show()


def plot2D(x, y):
    plt.plot(x, y)
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
    c1, c2, lbl = dataload.load2D("points_ring.pkl")
    disp2D2(c1, c2, lbl)
