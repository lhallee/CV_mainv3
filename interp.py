import numpy as np
import cv2
import scipy.interpolate as erp
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted
from glob import glob
from tqdm import tqdm


class interp_3d(object):
	def __init__(self, config):
		# Paths
		self.save_path = config.save_path
		self.points_path = config.points_path
		self.colors_path = config.colors_path
		self.img_paths = natsorted(glob(config.img_path + '*.png'))

		# Options
		self.rotate = config.rotate
		self.type = config.type
		self.mode = config.mode

		# Parameters
		self.s = config.scale
		self.d = config.density
		self.alpha = config.alpha

		if self.mode == 'save_npys':
			self.gen_points()
		elif self.mode == 'open_mesh' or 'open_vis' or 'open_ply':
			self.open_point_cloud()

	def gen_points(self):
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
		interpolation = erp.RegularGridInterpolator(grid, preds)
		print('Sparse Interpolation done')
		sZ = (Z - 1) * z_scale
		zF = (Z - 1) / self.d
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
		np.save(self.save_path + 'points.npy', points)
		np.save(self.save_path + 'colors.npy', colors)

	def open_point_cloud(self):
		import open3d
		points = np.load(self.points_path)
		colors = np.load(self.colors_path)
		pcd = open3d.geometry.PointCloud()
		pcd.points = open3d.utility.Vector3dVector(points)
		pcd.colors = open3d.utility.Vector3dVector(colors)

		if self.mode == 'open_vis':
			if self.rotate:
				open3d.visualization.draw_geometries_with_animation_callback([pcd], self.rotate_view)
			else:
				open3d.visualization.draw_geometries([pcd])

		elif self.mode == 'open_mesh':
			mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, self.alpha)
			mesh.compute_vertex_normals()
			if self.rotate:
				open3d.visualization.draw_geometries_with_animation_callback([mesh], self.rotate_view, mesh_show_back_face=True)
			else:
				open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

		elif self.mode == 'open_ply':
			open3d.io.write_point_cloud(
				self.save_path + str(self.s) + '_' + str(self.d) + str(self.type) + 'point_cloud.ply', pcd)

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
	# Paths
	parser.add_argument('--img_path', type=str, default='C:/Users/Logan/Desktop/full_result_square_1_26/')
	parser.add_argument('--save_path', type=str, default='./result/')
	parser.add_argument('--points_path', type=str, default='./result/points.npy')
	parser.add_argument('--colors_path', type=str, default='./result/colors.npy')

	# Options
	parser.add_argument('--rotate', type=bool, default=True)
	parser.add_argument('--type', type=str, default='hev')
	parser.add_argument('--mode', type=str, default='open_vis', help='Which task to perform: save_npys, open_mesh, open_vis, open_ply')

	# Parameters
	parser.add_argument('--scale', type=float, default=0.25)
	parser.add_argument('--density', type=float, default=1)
	parser.add_argument('--alpha', type=float, default=0.5)

	config = parser.parse_args()
	interp_3d(config)



