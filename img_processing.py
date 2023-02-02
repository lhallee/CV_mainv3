import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from glob import glob


#For making all imgs and GT images the same size with black boarders

file_path = './training_IMG_GT_1_23/'
save_path = './1_23_squared_results/'
img_paths = natsorted(glob(file_path + '*.png'))
GT_paths = natsorted(glob(file_path + '*.tif'))
max_H = 0
max_W = 0
dim = 0
heights = []
widths = []

for i in tqdm(range(len(img_paths))):
	img = np.array(cv2.imread(img_paths[i], 1))
	H, W, C = img.shape
	if H > max_H:
		max_H = H
	if W > max_W:
		max_W = W
	heights.append(H)
	widths.append(W)
if max_H >= max_W:
	dim = max_H
if max_W > max_H:
	dim = max_W

for i in tqdm(range(len(img_paths))):
	img = cv2.imread(img_paths[i], 1)
	old_H, old_W = heights[i], widths[i]
	dif_H, dif_W = dim - old_H, dim - old_W
	top = int(dif_H / 2)
	bottom = int(dif_H - top)
	right = int(dif_W / 2)
	left = int(dif_W - right)
	path = save_path + 'squared_' + img_paths[i].split('\\')[1] + '.png'
	new_img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
	cv2.imwrite(path, new_img)

max_H = 0
max_W = 0
dim = 0
heights = []
widths = []

for i in tqdm(range(len(GT_paths))):
	img = np.array(cv2.imread(GT_paths[i], 2))
	H, W = img.shape
	if H > max_H:
		max_H = H
	if W > max_W:
		max_W = W
	heights.append(H)
	widths.append(W)
if max_H >= max_W:
	dim = max_H
if max_W > max_H:
	dim = max_W

for i in tqdm(range(len(GT_paths))):
	img = cv2.imread(GT_paths[i], 2)
	old_H, old_W = heights[i], widths[i]
	dif_H, dif_W = dim - old_H, dim - old_W
	top = int(dif_H / 2)
	bottom = int(dif_H - top)
	right = int(dif_W / 2)
	left = int(dif_W - right)
	path = save_path + 'squared_' + GT_paths[i].split('\\')[1] + '.png'
	new_img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
	cv2.imwrite(path, new_img)

