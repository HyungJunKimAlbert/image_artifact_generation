#%% image check

import os
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("./lenna.png")
img = np.mean(img, axis=2, keepdims=True)
sz = img.shape
cmap = "gray" if sz[2] == 1 else None
print(sz)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")
plt.show()


# %%
