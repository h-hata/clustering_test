import numpy as np
from numpy.random import randn
import math
import pickle

n = 200
class_1 = 0.6 * randn(n, 2)
class_2 = 1.2 * randn(n, 2) + np.array([5, 1])
labels = np.hstack((np.ones(n), -np.ones(n)))
with open("points_normal_test.pkl", "wb") as f1:
    pickle.dump(class_1, f1)
    pickle.dump(class_2, f1)
    pickle.dump(labels, f1)

class_1 = 0.6 * randn(n, 2)
r = 0.8 * randn(n, 1) + 5
angle = 2 * math.pi * randn(n, 1)
class_2 = np.hstack((r * np.cos(angle), r * np.sin(angle)))
labels = np.hstack((np.ones(n), -np.ones(n)))
with open("points_ring_test.pkl", "wb") as f:
    pickle.dump(class_1, f)
    pickle.dump(class_2, f)
    pickle.dump(labels, f)
