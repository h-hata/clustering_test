import numpy as np
import dataload
import disp2D


class BaysClassifier(object):
    def __init__(self):
        self.lables = []
        self.mean = []
        self.var = []
        self.n = 0

    def train(self, data, labels=None):
        if labels is None:
            labels = range(len(data))
        self.labels = labels
        self.n = len(labels)
        for c in data:  # クラスごとに位置の平均分散を求める
            self.mean.append(np.mean(c, axis=0))
            self.var.append(np.var(c, axis=0))

    def classify(self, points):
        arr = [gauss(m, v, points) for m, v in zip(self.mean, self.var)]
        est_p = np.array(arr)
        ndx = est_p.argmax(axis=0)
        est_l = np.array([self.labels[n] for n in ndx])
        return est_p, est_l


def gauss(m, v, x):
    if len(x.shape) == 1:
        n, d = 1, x.shape[0]
    else:
        n, d = x.shape
    S = np.diag(1 / v)
    x = x - m
    y = np.exp(-0.5 * np.diag(np.dot(x, np.dot(S, x.T))))
    return y * (2 * np.pi) ** (-d / 2.0) / (np.sqrt(np.prod(v)) + 1e-6)


if __name__ == "__main__":
    # 教師学習データを読み込む
    c1, c2, labels = dataload.load2D("points_normal.pkl")
    model = BaysClassifier(labels, np.vstack((c1, c2)))
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
    disp2D.disp2D1(data, v)
