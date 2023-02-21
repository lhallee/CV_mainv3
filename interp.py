import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import open3d
import argparse
from natsort import natsorted
from glob import glob
from tqdm import tqdm


class interp_3d(object):
	def __init__(self, config):
		#paths
		self.save_path = config.save_path
		self.rotate = config.rotate
		self.img_paths = natsorted(glob(config.img_path + '*.png')[28:30])
		self.s = config.scale
		self.d = config.density
		self.type = config.type

	def run_interp(self):
		h, w = np.array(cv2.imread(self.img_paths[0], 2)).shape
		h, w = int(h * self.s), int(w / 2 * self.s)
		if self.type == 'hev':
			preds = np.concatenate(
				[np.array(cv2.resize(cv2.imread(self.img_paths[i], 2)[:, :5632], (h, w))).reshape(h, w, 1)
				 for i in tqdm(range(len(self.img_paths)))], axis=2)
		elif self.type =='lob':
			preds = np.concatenate(
				[np.array(cv2.resize(cv2.imread(self.img_paths[i], 2)[:, 5632:], (h, w))).reshape(h, w, 1)
				 for i in tqdm(range(len(self.img_paths)))], axis=2)
		X, Y, Z = preds.shape
		z_scale = 16 * self.s
		grid_x = np.arange(0, X, 1)
		grid_y = np.arange(0, Y, 1)
		grid_z = np.arange(0, Z, 1) * z_scale
		grid = (grid_x, grid_y, grid_z)
		print('Sparse grid done')
		interpolation = scipy.interpolate.RegularGridInterpolator(grid, preds)
		print('Sparse Interpolation done')
		sZ = (Z - 1) * z_scale
		zF = int(Z / self.d)
		print(X, Y, sZ, zF)
		interp_arr = np.mgrid[0:X - 1:self.d, 0:Y - 1:self.d, 0:sZ:(sZ/zF)].\
			reshape(3, int(((X - 1) / self.d) * ((Y - 1) / self.d) * zF)).T
		interp_grid = interp_arr.tolist()
		print('Dense grid done')
		interp_pred = np.array(interpolation(interp_grid))
		print('Dense Interpolation done')
		points = np.array(interp_grid)
		print(points.shape)
		zero_colors = np.zeros((len(interp_pred), 2))
		colors = np.hstack((interp_pred.reshape(len(interp_pred), 1), zero_colors)) / 255
		deletes = []
		for i in tqdm(range(len(colors)), desc='Removing Background'):
			if colors[i][0] < 0.5:
				deletes.append(i)

		points = np.delete(points, deletes, axis=0)
		colors = np.delete(colors, deletes, axis=0)
		pcd = open3d.geometry.PointCloud()
		pcd.points = open3d.utility.Vector3dVector(points)
		pcd.colors = open3d.utility.Vector3dVector(colors)
		if self.rotate:
			open3d.visualization.draw_geometries_with_animation_callback([pcd], self.rotate_view)
		else:
			open3d.visualization.draw_geometries([pcd])

	def plot4d(self, data, X, Y, z_scaled):
		# V3D = V.reshape(int((X-1)/c), int((Y-1)/c), int(sZ/c))
		data = data.reshape(int((X-1)/self.d), int((Y-1)/self.d), int(z_scaled/self.d))
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

	def rotate_view(self, vis):
		ctr = vis.get_view_control()
		ctr.rotate(10.0, 0.0)
		return False


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Model hyper-parameters
	parser.add_argument('--img_path', type=str, default='C:/Users/Logan Hallee/Desktop/1_26_result/result/')
	parser.add_argument('--save_path', type=str, default='./result/')
	parser.add_argument('--scale', type=float, default=0.01)
	parser.add_argument('--density', type=float, default=0.1)
	parser.add_argument('--rotate', type=bool, default=False)
	parser.add_argument('--type', type=str, default='hev')
	config = parser.parse_args()
	main = interp_3d(config)
	main.run_interp()



