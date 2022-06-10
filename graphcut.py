from tkinter import Image
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow
import bayes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import matplotlib.patches as patches

if __name__ == "__main__":
    # fname = "fish.jpg"
    fname = "person.png"
    im = np.array(Image.open(fname))

    print(im.shape)
    im = np.array(Image.fromarray(im).resize((50, 50), resample=Image.BILINEAR))

    plt.imshow(im)
    ax = plt.gca()
    rect = patches.Rectangle(
        (80, 10), 100, 100, linewidth=1, edgecolor="red", fill=False
    )
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (420, 360), 100, 100, linewidth=1, edgecolor="cyan", fill=False
    )
    ax.add_patch(rect)
    plt.show()
