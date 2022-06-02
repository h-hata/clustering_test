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

x = {}
x.setdefault("A", 0)
x["A"] = 7
x.setdefault("B", 0)
x["B"] = 6

print(max(x, key=lambda k: x[k]))


a = np.arange(12).reshape(3, 4)
print(a.shape)
print(a)
print(np.mean(a, axis=0))
print(np.mean(a, axis=1))
print("-------------")
range = [1, 5, 5, 8]
x = np.arange(range[0], range[1], 1)
y = np.arange(range[2], range[3], 1)
print(x, y)
xx, yy = np.meshgrid(x, y)
print(xx)
print(yy)

xxx = xx.ravel()
xxx2 = xx.flatten()
xx[0][0] = 100
print(xxx)

yyy2 = yy.flatten()
print(xxx2)
print(yyy2)
zzz = xxx2 + yyy2
print(zzz)
zz = zzz.reshape(xx.shape)
print(zz)

p = np.vstack((xxx2, yyy2))
print(p)

