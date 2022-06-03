import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import cm


# 2D Gauss Distribution Function
"""
x: 1 Dimension List of X, len=N
y: 1 Dimension List of Y, len=N
p[i]=(x[i],y[i])  0<=i<N
m: mean point [mean of x, mean of y]
s: covariance marix ( list of 2x2 dimension)
   s[0][0] varin

"""


def disp3D(xx, yy, zz):
    fig = plt.figure(figsize=(15, 5))  # unit inch

    # contour
    ax1 = fig.add_subplot(131, facecolor="w")
    ax1.set_aspect("equal")
    ax1.set_title("contour")
    ax1.contour(xx, yy, zz, np.arange(0.01, 0.14, 0.01))

    # surface
    ax2 = fig.add_subplot(132, projection="3d", facecolor="w")
    ax2.set_title("surface")
    # ax2.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax2.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap="plasma")
    # ax2.plot_surface(xx, yy, zz, rstride=1, cstride=1, color="green")

    # fireframe
    ax3 = fig.add_subplot(133, projection="3d", facecolor="w")
    ax3.set_title("wireframe")
    ax3.plot_wireframe(
        xx, yy, zz, linewidth=0.5, color="green",
    )

    plt.show()


def gauss2D(p, m, v):
    det = np.linalg.det(v)
    try:
        inv = np.linalg.inv(v)
    except np.linalg.LinAlgError:
        return np.zeros(0)
    X = (p - m) @ inv @ (p - m).T
    XX = -np.diag(X)
    XXX = np.exp(XX / 2.0)
    denom = np.sqrt((2 * np.pi) ** 2 * det + 1e-9)  # add small to denom. for safe
    y = XXX / denom
    return y


# range of calclation
x = y = np.arange(-4, 4, 0.2)  # 40x40 points mesh
xx, yy = np.meshgrid(x, y)
xxx = xx.flatten()
yyy = yy.flatten()
p = np.c_[xxx, yyy]
m = np.array([0, 0])  # mean
v = np.array([[1, 0.3], [0.3, 2]])  # covariance matrix
zzz = gauss2D(p, m, v)
if zzz.shape[0] != p.shape[0]:
    print("Error")
    exit(0)
zz = zzz.reshape(xx.shape)
disp3D(xx, yy, zz)

