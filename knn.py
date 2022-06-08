import dataload
import math
import numpy as np
import disp2D


class KnnClassifier(object):
    def __init__(self, labels, samples):
        self.lables = labels
        self.samples = samples
        self.labelset = set(labels)

    def classify(self, point, k=3):
        dist = np.array([L2dist(point, s) for s in self.samples])
        # print(dist)
        ndx = dist.argsort()
        votes = {}
        for lbl in self.labelset:
            votes.setdefault(lbl, 0)
        for i in range(k):
            label = self.lables[ndx[i]]
            votes[label] += 1
        return max(votes, key=lambda x: votes[x])


def L2dist(p1, p2):
    return math.sqrt(sum((p1 - p2) ** 2))


if __name__ == "__main__":
    # 教師学習データを読み込む
    c1, c2, labels = dataload.load2D("points_normal.pkl")
    model = KnnClassifier(labels, np.vstack((c1, c2)))
    # テストデータを読み込む
    c1, c2, labels = dataload.load2D("points_normal_test.pkl")
    data = np.vstack((c1, c2))
    # 一点づつ分類する
    judge = []
    for p in data:
        judge.append(model.classify(p))
    # 検証
    v = []
    for i in range(len(data)):
        if labels[i] != judge[i]:
            v.append(-1)
        else:
            v.append(1)

    # XY全体で分類を行う

    x = np.arange(-2, 9, 0.2)
    y = np.arange(-3, 4, 0.2)
    xx, yy = np.meshgrid(x, y)
    xxx = xx.ravel()
    yyy = yy.ravel()
    pp = np.c_[xxx, yyy]
    z = np.empty(0)
    for p in pp:
        j = model.classify(p)
        z = np.append(z, j)
    zz = z.reshape(xx.shape)
    disp2D.disp2D1WithContour(data, labels, xx, yy, zz, 0)
