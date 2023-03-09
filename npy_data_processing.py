import argparse
import numpy as np
import torch
import torchvision
import cv2
from natsort import natsorted
from skimage.util import view_as_windows
from tqdm import tqdm
from glob import glob

class training_processing:
    def __init__(self, config):
        self.train_img_path = config.train_img_path
        self.val_img_path = config.val_img_path
        self.GT_paths = config.GT_paths
        self.dim = config.image_size
        self.num_class = config.num_class

    def crop_augment_GTs(self, GT):
        GTs = view_as_windows(GT, (self.dim, self.dim, self.num_class), step=self.dim)
        a, b, c, d, e, f = GTs.shape
        GTs = GTs.reshape(a * b, self.dim, self.dim, self.num_class)
        GTs[GTs < 1] = 0 #GT pixels should have 0 background
        GTs[GTs > 0] = 1 #and actual features 1
        GTs_90 = np.copy(GTs) #same copies for GT
        GTs_vflip = np.copy(GTs)
        GTs_hflip = np.copy(GTs)
        GTs_jitter_1 = np.copy(GTs)
        for i in range(len(GTs)):
            GTs_90[i] = np.rot90(GTs_90[i])
            GTs_vflip[i] = np.flipud(GTs_vflip[i])
            GTs_hflip[i] = np.fliplr(GTs_hflip[i])
        final_crops_GT = np.concatenate((GTs, GTs_90, GTs_hflip, GTs_vflip, GTs_jitter_1))
        return final_crops_GT

    def crop_augment_imgs(self, img):
        img = np.array(cv2.imread(img, 1)) / 255.0 #read and scale img
        imgs = view_as_windows(img, (self.dim, self.dim, 3), step=self.dim)
        a, b, c, d, e, f = imgs.shape
        imgs = imgs.reshape(a * b, self.dim, self.dim, 3) #reshape windowed output into num_images, dim, dim channel
        #Augmentations
        imgs_90 = np.copy(imgs)
        imgs_vflip = np.copy(imgs)
        imgs_hflip = np.copy(imgs)
        #reshape for torch augmentations
        imgs_jitter_1 = torch.tensor(np.transpose(np.copy(imgs), axes=(0, 3, 1, 2)), dtype=torch.float32)
        for i in range(len(imgs)):
            imgs_90[i] = np.rot90(imgs_90[i])
            imgs_vflip[i] = np.flipud(imgs_vflip[i])
            imgs_hflip[i] = np.fliplr(imgs_hflip[i])
            #perform various jitter augmentations with a new probability each time
            transform_1 = torchvision.transforms.ColorJitter(np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5),
                                                           np.random.uniform(0.0, 0.5))
            imgs_jitter_1[i] = transform_1(imgs_jitter_1[i])

        imgs_jitter_1 = np.transpose(np.array(imgs_jitter_1), axes=(0, 2, 3, 1)) #reshape back to normal
        final_crops = np.concatenate((imgs, imgs_90, imgs_hflip, imgs_vflip, imgs_jitter_1)) #combine all together
        return final_crops

    def crop_val(self, img):
        img = np.array(cv2.imread(img, 1)) / 255.0  # load and scale img
        imgs = view_as_windows(img, (self.dim, self.dim, 3), step=int(self.dim / 2))
        a, b, c, d, e, f = imgs.shape
        imgs = imgs.reshape(a * b, self.dim, self.dim, 3)
        return imgs

    def to_large_dataarray(self):
        train_img_paths = natsorted(glob(self.train_img_path + '*')) # natural sort
        val_img_paths = natsorted(glob(self.val_img_path + '*'))
        GTs = []
        for i in range(self.num_class):
            GTs.append(natsorted(glob(self.GT_paths[i] + '*')))

        assert len(train_img_paths) * self.num_class == sum([len(GTs[i]) for i in range(len(GTs))]),\
            'Need GT for every Image.'

        #Combine results from each image path into one array
        crop_imgs = np.concatenate([self.crop_augment_imgs(img=train_img_paths[i])
                                    for i in tqdm(range(len(train_img_paths)), desc='Cropping Images')], axis=0)
        #Read the GT images and stack all the classes together in this form (n, h, w, num_class)
        read_GT = np.concatenate([
            np.concatenate([np.array(cv2.imread(GTs[i][j], 2), dtype=np.float32).reshape(1,
                                                                    np.array(cv2.imread(GTs[i][j], 2)).shape[0],
                                                                    np.array(cv2.imread(GTs[i][j], 2)).shape[1], 1)
                            for j in range(len(GTs[i]))], axis=0)
            for i in range(len(GTs))], axis=3)

        crop_GTs = np.concatenate([self.crop_augment_GTs(GT=read_GT[i])
                                   for i in tqdm(range(len(read_GT)), desc='Cropping GTs')], axis=0)

        val_imgs = np.concatenate([self.crop_val(img=val_img_paths[i])
                                   for i in tqdm(range(len(val_img_paths)), desc='Cropping Val')], axis=0)
        #Trim some all black images
        print(crop_imgs.shape, crop_GTs.shape)
        dels = []
        for i in range(len(crop_imgs)):
            if np.count_nonzero(crop_imgs[i]) <= 0.01 * crop_imgs[i].size and i % 2 == 0:
                dels.append(i)
        crop_imgs = np.delete(crop_imgs, dels, axis=0)
        crop_GTs = np.delete(crop_GTs, dels, axis=0)
        print(crop_imgs.shape, crop_GTs.shape)
        #numpy array to torch tensor, move around columns for pytorch convolution
        crop_imgs = np.transpose(crop_imgs, axes=(0, 3, 1, 2))
        crop_GTs = np.transpose(crop_GTs, axes=(0, 3, 1, 2))
        val_imgs = np.transpose(val_imgs, axes=(0, 3, 1, 2))

        return crop_imgs, crop_GTs, val_imgs

def main(config):
    if config.mode == 'eval':
        print('Eval')
        #eval mode is for evaluating the 3D reconstruction capabilities of a model
        #data_setup = eval_processing(config)
        #eval_loader, num_col, num_row = data_setup.eval_dataloader()
        #dataloader of eval data, number of columns in window split, number of rows in window split
        #solver = eval_solver(config, eval_loader, num_col, num_row)
        #solver.eval()

    #Can choose between real data in a path or generated data of squares of various sizes
    elif config.mode == 'train':
        data_setup = training_processing(config)
        crop_imgs, crop_GTs, val_imgs = data_setup.to_large_dataarray()
        mini_imgs, mini_GTs = crop_imgs[1000:2000], crop_GTs[1000:2000]
        np.save(config.save_path + str(config.image_size) + 'img_data.npy', crop_imgs)
        np.save(config.save_path + str(config.image_size) + 'GT_data.npy', crop_GTs)
        np.save(config.save_path + str(config.image_size) + 'val_img_data.npy', val_imgs)
        np.save(config.save_path + str(config.image_size) + 'mini_img.npy', mini_imgs)
        np.save(config.save_path + str(config.image_size) + 'mini_GT.npy', mini_GTs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for segmentation')
    # Paths
    parser.add_argument('--train_img_path', type=str, default='./img_data/train_img/',
                        help='Path for training images')
    parser.add_argument('--val_img_path', type=str, default='./img_data/val_img/',
                        help='Path for validation images')
    parser.add_argument('--GT_paths', type=list, default=['./img_data/GT_1/', './img_data/GT_2/'],
                        help='List of paths for training GT (one for each class)')
    parser.add_argument('--eval_img_path', type=str, default='./img_data/eval_img/',
                        help='Images for 2D reconstruction evaluation')
    parser.add_argument('--save_path', type=str, default='./processed_data/')
    # misc
    parser.add_argument('--mode', type=str, default='train', help='train, eval, CV')
    config = parser.parse_args()
    main(config)