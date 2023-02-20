import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
from scipy.ndimage import rotate


def align_images(im1, im2):
    # Convert images to grayscale
    # Find size of image1
    sz = im1.shape
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN
    # Define the warp matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define the number of iterations
    number_of_iterations = 500
    # Define correlation coefficient threshold
    # Specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)
    # Use warpPerspective for Homography
    im_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im_aligned, warp_matrix

'''
top, bottom, left, right = 500, 500, 500, 500


temp = cv2.copyMakeBorder(np.array(cv2.imread('img.png', 2), dtype=np.float32),
                          top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

img = rotate(cv2.copyMakeBorder(np.array(cv2.imread('img.png', 2), dtype=np.float32),
                          top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0),
             30, reshape=False)

scale_percent = 10  # percent of original size
width = int(temp.shape[1] * scale_percent / 100)
height = int(temp.shape[0] * scale_percent / 100)
dim = (width, height)
im1 = cv2.resize(temp, dim)
im2 = cv2.resize(img, dim)

align, warp_matrix = align_images(im1, im2)
'''

stacks_path = 'C:/Users/Logan Hallee/Desktop/1_26_result/result/'
lop = natsorted(glob(stacks_path + 'sec*'))
print(lop)
full_paths = []
for i in range(len(lop)):
    full_paths.append(natsorted(glob(lop[i] + '/*.png')))

for i in range(len(full_paths)-1):
    template = np.array(cv2.imread(full_paths[i][-1], 2)[:, 5632:], dtype=np.float32)
    to_rot = np.array(cv2.imread(full_paths[i+1][0], 2)[:, 5632:], dtype=np.float32)
    big_sz = template.shape
    scale_percent = 10  # percent of original size
    width = int(big_sz[1] * scale_percent / 100)
    height = int(big_sz[0] * scale_percent / 100)
    dim = (width, height)
    im1 = cv2.resize(template, dim)
    sz = im1.shape
    im2 = cv2.resize(to_rot, dim)
    aligned, warp_matrix = align_images(im1, im2)
    for i in range(len(full_paths[i+1])):
        im_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
