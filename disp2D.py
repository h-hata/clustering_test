import matplotlib.pyplot as plt
import dataload


def disp2D(data1, data2, labels):
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
    disp2D(c1, c2, l)

