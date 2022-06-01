import numpy as np
import math


def L2dist(p1, p2):
    print(sum((p1 - p2) ** 2))
    return math.sqrt(sum((p1 - p2) ** 2))


x = np.array([[2, 3], [4, 2], [-1, 2]])
print(x)
print(L2dist(x[0], x[1]))

x = [i * 2 for i in range(3)]
print(x)
y = []
for i in range(3):
    y.append(i * i)
print(y)
