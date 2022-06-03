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
        return est_l, est_p

    def classify_bak(self, points):
        arr = [gauss_bak(m, v, points) for m, v in zip(self.mean, self.var)]
        est_p = np.array(arr)
        ndx = est_p.argmax(axis=0)
        est_l = np.array([self.labels[n] for n in ndx])
        return est_l, est_p


def gauss(m, v, x):
    if len(x.shape) == 1:
        n, d = 1, x.shape[0]
    else:
        n, d = x.shape
    S = np.diag(1 / v)
    x = x - m
    y = np.exp(-0.5 * np.diag(x @ S @ x.T))
    return y * (2 * np.pi) ** (-d / 2.0) / (np.sqrt(np.prod(v)) + 1e-6)


def gauss_bak(m, v, x):
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
    bc = BaysClassifier()
    bc.train([c1, c2], [1, -1])

    # テストデータを読み込む
    c1, c2, labels = dataload.load2D("points_normal_test.pkl")
    data = np.vstack((c1, c2))
    v = bc.classify(data)
    v2 = bc.classify_bak(data)
    if np.all(v != v2):
        print("Unmatch")
    disp2D.disp2D1(data, v[0])

