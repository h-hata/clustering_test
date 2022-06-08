import matplotlib.pyplot as plt
import numpy as np


def step(p):
    if p[0] < 0.5:
        val = 0
    elif p[1] < 0.5:
        val = 2
    else:
        val = 3
    return val


def cone(p):
    return 3 - np.sqrt(p[0] ** 2 + 2 * p[1] ** 2 - 2 * p[0] * p[1])


def plane(p):
    return p[0] - 2 * p[1] + 1


def plane2(p):
    return -p[0] + p[1] - 1


def f(p):
    return


x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
xx, yy = np.meshgrid(x, y)
xxx = xx.ravel()
yyy = yy.ravel()
p = np.c_[xxx, yyy]
z = np.array([*map(lambda x: step(x), p)])
zz = z.reshape(xx.shape)
fig = plt.figure(figsize=(8, 8))  # unit inch
ax1 = fig.add_subplot(221, projection="3d", facecolor="w")
ax1.set_title("surface")
ax1.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap="plasma")
ax2 = fig.add_subplot(222, projection="3d", facecolor="w")
ax2.set_title("wireframe")
ax2.plot_wireframe(
    xx, yy, zz, linewidth=0.5, color="green",
)
ax3 = fig.add_subplot(223, projection="3d", facecolor="w")
ax3.set_title("contour")
ax3.contour(xx, yy, zz, [2.5])
ax4 = fig.add_subplot(224, facecolor="w")
ax4.set_aspect("equal")
ax4.set_title("contour")
ax4.contour(xx, yy, zz, [2.5])
plt.show()

