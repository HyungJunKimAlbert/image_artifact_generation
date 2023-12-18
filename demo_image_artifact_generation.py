import os
import numpy as np
import matplotlib.pyplot as plt

#%% image check
img = plt.imread("./lenna.png")
sz = img.shape
cmap = "gray" if sz[2] == 1 else None

plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")
plt.show()

