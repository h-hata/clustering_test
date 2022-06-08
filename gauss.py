import numpy as np
import disp2D


# 1th-D Gauss Distribution Function
def gauss1D(x, m, v):
    X = (x - m) ** 2
    XX = -X / v
    XXX = np.exp(XX / 2.0)
    denom = np.sqrt(2 * np.pi * v + 1e-9)  # add small to denom. for safty
    y = XXX / denom
    return y


# Nth-D Gauss Distribution Function
def gaussN(p, m, v):
    det = np.linalg.det(v)
    try:
        inv = np.linalg.inv(v)
    except np.linalg.LinAlgError:
        return np.zeros(0)
    n = v.shape[0]
    X = (p - m) @ inv @ (p - m).T
    XX = -np.diag(X)
    XXX = np.exp(XX / 2.0)
    # add small to denom. for safty
    denom = np.sqrt((2 * np.pi) ** n * det + 1e-9)
    y = XXX / denom
    return y


"""
x: 1 Dimension List of X, len=N
y: 1 Dimension List of Y, len=N
p[i]=(x[i],y[i])  0<=i<N
m: mean point [mean of x, mean of y]
s: covariance marix ( list of 2x2 dimension)
   s[0][0] varin
"""


def gauss(p, m, v):
    if type(v) is not np.ndarray:
        return gauss1D(p, m, v)
    else:
        return gaussN(p, m, v)


def test2D():
    # range of calclation
    x = y = np.arange(-4, 4, 0.2)  # 40x40 points mesh
    xx, yy = np.meshgrid(x, y)
    xxx = xx.flatten()
    yyy = yy.flatten()
    p = np.c_[xxx, yyy]
    m = np.array([0, 0])  # mean
    v = np.array([[1, 0], [0, 4]])  # covariance matrix
    zzz = gauss(p, m, v)
    if zzz.shape[0] != p.shape[0]:
        print("Error")
        exit(0)
    zz = zzz.reshape(xx.shape)
    disp2D.disp3D(xx, yy, zz)
    return  # for break point


def test1D():
    x = np.arange(-5, 5, 0.2)  # 40points
    m = 0
    v = 4
    z = gauss(x, m, v)
    disp2D.plot2D(x, z)


if __name__ == "__main__":
    test2D()
    exit(0)
