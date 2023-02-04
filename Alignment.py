import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def align_images(im1, im2):
    # Convert images to grayscale
    scale_percent = 10  # percent of original size
    width = int(im1.shape[1] * scale_percent / 100)
    height = int(im1.shape[0] * scale_percent / 100)
    dim = (width, height)
    im1_scale = cv2.resize(im1, dim)
    im2_scale = cv2.resize(im2, dim)
    print(im2_scale.shape)
    # Find size of image1
    sz = im1_scale.shape

    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Define the warp matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define the number of iterations
    number_of_iterations = 20000
    # Define correlation coefficient threshold
    # Specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_scale, im2_scale, warp_matrix, warp_mode, criteria)

    # Use warpPerspective for Homography
    im_aligned = cv2.warpAffine(im2_scale, warp_matrix, (sz[1], sz[0]),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im_aligned, warp_matrix

top, bottom, left, right = 500, 500, 500, 500


temp = cv2.copyMakeBorder(np.array(cv2.imread('img.png', 2))[:, 3584:],
                          top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

img = rotate(cv2.copyMakeBorder(np.array(cv2.imread('img.png', 2))[:, 3584:],
                          top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0),
             60, reshape=False)

align, warp_matrix = align_images(temp, img)


plt.imshow(temp)
plt.show()
plt.imshow(img)
plt.show()
plt.imshow(align)
plt.show()
plt.imshow(cv2.warpAffine(img, warp_matrix, (img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
plt.show()