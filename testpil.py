from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

fname = "fish.jpg"
fname2 = "person.png"
pil_im = Image.open(fname2)
im = np.array(pil_im)
print(im.shape, im.dtype)

"""
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]
plt.imshow(im)
plt.plot(x, y, "rs-")
##plt.plot(x[:2], y[:2])
"""
"""
im2 = np.array(pil_im.convert("L"))
plt.imshow(im2, cmap="gray")
plt.show()
region = pil_im.crop((400, 100, 700, 400))
im = np.array(region.rotate(45))
print(im.shape, im.dtype)
plt.imshow(im)
"""
plt.show()

