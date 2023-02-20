import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import open3d
from natsort import natsorted
from glob import glob
from tqdm import tqdm

file_path = 'C:/Users/Logan Hallee/Desktop/1_26_result/result/'
save_path = './result/'
img_paths = natsorted(glob(file_path + '*.png'))

r = 0.2
h, w = np.array(cv2.imread(img_paths[0], 2)).shape
h, w = int(h * r), int(w / 2 * r)
preds = np.concatenate([np.array(cv2.resize(cv2.imread(img_paths[i], 2)[:, :5632], (h, w))).reshape(h, w, 1)
                        for i in tqdm(range(len(img_paths)))], axis=2)
# plt.imshow(preds[:,:,0])
# plt.show()

X, Y, Z = preds.shape

s = 16 * r
grid_x = np.arange(0, X, 1)
grid_y = np.arange(0, Y, 1)
grid_z = np.arange(0, Z, 1) * s
grid = (grid_x, grid_y, grid_z)

print('Sparse grid done')
interpolation = scipy.interpolate.RegularGridInterpolator(grid, preds)
print('Sparse Interpolation done')
c = 1
sZ = (Z - 1) * s
sc = int(Z / c)
# assert sZ % c == 0, 'Voxel scales and z stack scale must be divisible for mgrid to work'
print(X, Y, sZ, sc)
interp_arr = np.mgrid[0:X - 1:c, 0:Y - 1:c, 0:sZ:(sZ / sc)].reshape(3, int(((X - 1) / c) * ((Y - 1) / c) * sc)).T
interp_grid = interp_arr.tolist()
print('Dense grid done')
V = np.array(interpolation(interp_grid))
print('Dense Interpolation done')
points = np.array(interp_grid)
print(points.shape)
zero_colors = np.zeros((len(V), 2))
colors = np.hstack((V.reshape(len(V), 1), zero_colors)) / 255

deletes = []
for i in tqdm(range(len(colors)), desc='Removing Background'):
	if colors[i][0] < 0.5:
		deletes.append(i)

points = np.delete(points, deletes, axis=0)
colors = np.delete(colors, deletes, axis=0)
# V3D = V.reshape(int((X-1)/c), int((Y-1)/c), int(sZ/c))

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(points)
pcd.colors = open3d.utility.Vector3dVector(colors)


# mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.03)
# mesh.compute_vertex_normals()
# open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
open3d.visualization.draw_geometries([pcd])
def rotate_view(vis):
	ctr = vis.get_view_control()
	ctr.rotate(10.0, 0.0)
	return False

#open3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)


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
# plt.show()

# plot4d(V3D)
