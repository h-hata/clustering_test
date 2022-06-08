import numpy as np
import dataload
import disp2D
import gauss


class BaysClassifier(object):
    def __init__(self):
        self.lables = []
        self.mean = []
        self.var = [[]]
        self.n = 0

    def train(self, data, labels=None):
        if labels is None:
            labels = range(len(data))
        self.labels = labels
        self.n = len(labels)
        for c in data:  # クラスごとに位置の平均分散を求める
            mu=np.mean(c, axis=0)
            cov = np.cov(c)
            continue
        return

    def classify(self, points):
        arr = [gauss.gauss(m, v, points) for m, v in zip(self.mean, self.var)]
        est_p = np.array(arr)
        ndx = est_p.argmax(axis=0)
        est_l = np.array([self.labels[n] for n in ndx])
        return est_l, est_p


if __name__ == "__main__":
    # 教師学習データを読み込む
    c1, c2, labels = dataload.load2D("points_normal.pkl")
    bc = BaysClassifier()
    bc.train([c1, c2], [1, -1])

    # テストデータを読み込む
    c1, c2, labels = dataload.load2D("points_normal_test.pkl")
    data = np.vstack((c1, c2))
    v = bc.classify(data)

    if np.all(v != v2):
        print("Unmatch")
    disp2D.disp2D1(data, v[0])

