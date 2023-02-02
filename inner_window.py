import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import view_as_windows

dim = 512
step = 512
file = 'C:/Users/Logan Hallee/Desktop/Segmentation/CV_mainv2/eval_img/squared_section 5_z1c1+2+3.png'
test = np.array(cv2.imread(file, 1))
print(test.shape)
def window_recon(SR, num_col, num_row, dim, step):
	# Reconstruct HEV and LOB from multiclass segmentation model
	hstep = int(step / 2)
	recon = np.zeros(((num_col+1) * dim, (num_row+1) * dim))
	k = 0
	for i in range(num_col):
		for j in range(num_row):
			inner = SR[k][hstep:3*hstep, hstep:3*hstep]
			recon[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = inner
			k += 1
	return recon


def crop_recon(img, dim, step):
	#a is num_col, b is num_row
	imgs = view_as_windows(img, (dim, dim, 3), step=step)
	a, b, c, d, e, f = imgs.shape
	imgs = imgs.reshape(a * b, dim, dim, 3)
	return imgs, a, b

imgs, num_col, num_row = crop_recon(test, 2*dim, step)
print(imgs.shape)
recon = window_recon(imgs[:,:,:,2], num_col, num_row, dim, step)
print(recon.shape)
plt.imshow(recon)
plt.show()

