import dataload
import math
import numpy as np


class KnnClassifier(object):
    def __init__(self, labels, samples):
        self.lables = labels
        self.samples = samples

    def classify(self, point, k=3):
        dist = np.array([L2dist(point, s) for s in self.samples])
        print(dist)
        ndx = dist.argsort()
        votes = {}
        for i in range(k):
            label = self.lables[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1
        return max(votes, key=lambda x: votes.get(x))


def L2dist(p1, p2):
    return math.sqrt(sum((p1 - p2) ** 2))


if __name__ == "__main__":
    c1, c2, labels = dataload.load2D("points_normal.pkl")
    model = KnnClassifier(labels, np.vstack((c1, c2)))
    print(model.classify(c1[0]))
