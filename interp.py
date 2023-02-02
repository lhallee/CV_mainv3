import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
from tqdm import tqdm


file_path = 'C:/Users/Logan Hallee/Desktop/1_26_result/result/'
save_path = './result/'
img_paths = natsorted(glob(file_path + '*.png'))

d = 16
r = 0.01
h, w = np.array(cv2.imread(img_paths[0], 2)).shape
h, w = int(h * r), int(w/2 * r)
preds = np.concatenate([np.array(cv2.resize(cv2.imread(img_paths[i], 2), (h, w))).T.reshape(h, w, 1)
                        for i in tqdm(range(len(img_paths)))], axis=2)

#plt.imshow(preds[:,:,9])
#plt.show()

X, Y, Z = preds.shape

s = 16
grid_x = np.arange(0, X, 1)
grid_y = np.arange(0, Y, 1)
grid_z = np.arange(0, Z, 1) * s
grid = (grid_x, grid_y, grid_z)

print('Sparse grid done')
interpolation = scipy.interpolate.RegularGridInterpolator(grid, preds)
print('Sparse Interpolation done')
c = 0.5
sZ = (Z-1) * s
interp_arr = np.mgrid[0:X-1:c, 0:Y-1:c, 0:sZ:c].reshape(3, int(((X-1)/c)*((Y-1)/c)*(sZ/c))).T
interp_grid = interp_arr.tolist()
print('Dense grid done')
V = interpolation(interp_grid)
print('Dense Interpolation done')
V3D = V.reshape(int((X-1)/c), int((Y-1)/c), int(sZ/c))


def plot4d(data):
	print('Plotting')
	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(projection="3d")
	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False
	mask = data > 0.9
	idx = np.arange(int(np.prod(data.shape)))
	x, y, z = np.unravel_index(idx, data.shape)
	ax.scatter(x, y, z, c=data.flatten(), s=10.0 * mask, edgecolor='face', alpha=0.01, marker='o', cmap='Reds',
	           linewidth=0)
	plt.tight_layout()
	plt.savefig("test_scatter_4d.png", dpi=250)
	print('Saved')
	#plt.show()

plot4d(V3D)

