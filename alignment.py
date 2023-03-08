import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
from tqdm import tqdm


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


def align_stacks(stacks_path, save_path):
    sections = natsorted(glob(stacks_path + 'sec*')) # predicted sections
    al_sections, full_paths = [], [] # sections for aligned results, all predicted images
    # makes paths in save_path with how many sections needed
    for i in tqdm(range(len(sections)), desc='Create Directories'):
        path = save_path + 'section' + str(i+1) + '/'
        if not os.path.exists(path):
            os.mkdir(path)
        al_sections.append(path)
    for i in range(len(sections)):
        full_paths.append(natsorted(glob(sections[i] + '/*.png')))
    # full_paths has all sections and their image paths
    # write section 1 to save_path, doesn't get transformed
    for i in tqdm(range(len(full_paths[0])), desc='Save Section 1'):
        plt.imsave(al_sections[0] + full_paths[0][i].split('\\')[-1],
                   np.array(cv2.imread(full_paths[0][i], 2))[:, 5632:])
    # now section1 images are in al_sections[0] directory
    # need to transform full_paths[i] into al_sections[i] using al_sections[i-1][-1] and full_paths[i][0]
    for i in tqdm(range(len(full_paths)-1), desc='Aligning'):
        # add saved images to full_al_paths, similar to full_paths, but every loop so it's updated
        full_al_paths = []
        for k in range(len(al_sections)-1):
            full_al_paths.append(natsorted(glob(al_sections[k] + '/*.png')))
        template = np.array(cv2.imread(full_al_paths[i][-1], 2), dtype=np.float32)
        to_rot = np.array(cv2.imread(full_paths[i + 1][0], 2)[:, 5632:], dtype=np.float32)
        sz = template.shape
        aligned, warp_matrix = align_images(template, to_rot)
        for j in range(len(full_paths[i + 1])):
            to_rot = np.array(cv2.imread(full_paths[i + 1][j], 2), dtype=np.float32)[:, 5632:]
            im_aligned = cv2.warpAffine(to_rot, warp_matrix, (sz[1], sz[0]),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            plt.imsave(al_sections[i + 1] + full_paths[i + 1][j].split('\\')[-1], im_aligned)


stacks_path = 'C:/Users/Logan Hallee/Desktop/LN IMAGES/1_26_sectioned_results/'
save_path = 'C:/Users/Logan Hallee/Desktop/LN IMAGES/1_26aligned/'
align_stacks(stacks_path, save_path)

'''
        scale_percent = 100  # percent of original size
        width = int(big_sz[1] * scale_percent / 100)
        height = int(big_sz[0] * scale_percent / 100)
'''
