#%% 
import os
import numpy as np
import matplotlib.pyplot as plt

# image check
img = plt.imread("./lenna.png")
img = np.mean(img, axis=2, keepdims=True)
sz = img.shape
cmap = "gray" if sz[2] == 1 else None
print(sz)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")
plt.show()


#%% 1-1. Inpainting: Uniform sampling
ds_y = 2
ds_x = 4
msk = np.zeros(sz)
msk[::ds_y, ::ds_x, :] = 1

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('Ground Truth')

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title('Uniform sampling mask')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('Sampling Image')



#%% 1-2. Inpainting: Random Sampling
# img = plt.imread("./lenna.png")
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# prob = 0.5

img = plt.imread("./lenna.png")
rnd = np.random.rand(sz[0], sz[1], 1)
prob = 0.5

msk = (rnd > prob).astype(float)
dst = img * msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('Ground Truth')

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title('Random Mask')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('Sampling Image')

#%% 1-3. Inpainting: Gaussian sampling
ly = np.linspace(-1, 1, sz[0])
lx = np.linspace(-1, 1, sz[1])

x, y = np.meshgrid(lx, ly)

x0 = 0
y0 = 0
sgmx = 1
sgmy = 1

a = 1
gaus = a*np.exp(-((x-x0)**2/(2*sgmx**2) + (y-y0) **2 / (2*sgmy**2))) # 2d gaussian distribution
gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
rnd = np.random.rand(sz[0], sz[1], sz[2])
# plt.imshow(gaus)
msk = (rnd < gaus).astype(float)
dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('Ground Truth')

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling Image")
# %%
