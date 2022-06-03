import numpy as np
import math

x = []
y = []
for _ in np.arange(1000):
    n1 = math.floor(np.random.random() * 2)
    n2 = math.floor(np.random.random() * 2)
    x.append(n1)
    y.append(n1 + n2)
p = np.vstack((x, y))
p = p.T
a = np.zeros((2, 3))
for i in np.arange(len(x)):
    a[x[i]][y[i]] += 1
print(a)
