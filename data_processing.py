import numpy as np
import os
import torch
import torchvision
import cv2
from natsort import natsorted
from torch.utils import data
from skimage.util import view_as_windows
from tqdm import tqdm
from glob import glob

class ImageSet(data.Dataset):
    # Custom pytorch dataset, simply indexes imgs and gts
    def __init__(self, imgs, GTs):
        self.imgs = imgs
        self.GTs = GTs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = torch.tensor(self.imgs[index], dtype=torch.float)
        GT = torch.tensor(self.GTs[index], dtype=torch.float)
        return img, GT

class ReconSet(data.Dataset):
    # Custom pytorch dataset, simply indexes imgs
    def __init__(self, imgs):
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = torch.tensor(self.imgs[index], dtype=torch.float)
        return img


class training_processing:
    def __init__(self, config):
        self.train_img_path = config.train_img_path
        self.val_img_path = config.val_img_path
        self.val_GT_path = config.val_GT_paths
        self.GT_paths = config.GT_paths
        self.dim = config.image_size
        self.num_class = config.num_class
        self.batch_size = config.batch_size
        self.num_cpu = os.cpu_count()

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

    def to_dataloader(self):
        train_img_paths = natsorted(glob(self.train_img_path + '*'))[:1] # natural sort
        val_img_paths = natsorted(glob(self.val_img_path + '*'))[:1]
        GTs = []
        for i in range(self.num_class):
            GTs.append(natsorted(glob(self.GT_paths[i] + '*'))[:1])

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


        #numpy array to torch tensor, move around columns for pytorch convolution
        crop_imgs = np.transpose(crop_imgs, axes=(0, 3, 1, 2))
        crop_GTs = np.transpose(crop_GTs, axes=(0, 3, 1, 2))
        val_imgs = np.transpose(val_imgs, axes=(0, 3, 1, 2))


        train_data = ImageSet(crop_imgs, crop_GTs)  # move to pytorch dataset
        val_data = ReconSet(val_imgs)

        train_loader = data.DataLoader(train_data, batch_size=self.batch_size,
                                       shuffle=True, drop_last=True, num_workers=self.num_cpu)

        val_loader = data.DataLoader(val_data, batch_size=self.batch_size,
                                     shuffle=False, drop_last=False, num_workers=self.num_cpu)

        return train_loader, val_loader


class eval_processing:
    def __init__(self, config):
        self.eval_path = config.eval_img_path
        self.eval_type = config.eval_type
        self.dim = config.image_size
        self.num_class = config.num_class
        self.batch_size = config.batch_size
        self.num_cpu = os.cpu_count()

    def load_imgs(self):
        eval_paths = natsorted(glob(self.eval_path + '*.png'))  # natural sort
        return eval_paths
    def crop_recon(self, img):
        img = np.array(cv2.imread(img, 1)) / 255.0 #load and scale img
        imgs = view_as_windows(img, (2*self.dim, 2*self.dim, 3), step=int(self.dim/2))
        a, b, c, d, e, f = imgs.shape
        imgs = imgs.reshape(a*b, 2*self.dim, 2*self.dim, 3)
        imgs = np.transpose(imgs, axes=(0, 3, 1, 2))
        return imgs, a, b

    def eval_dataloader(self):
        eval_paths = self.load_imgs()
        if self.eval_type == 'Windowed':
            #path to crop_recon, concatenate results
            window_imgs = np.concatenate([self.crop_recon(eval_paths[i])[0]
                                          for i in tqdm(range(len(eval_paths)), desc='Crop Images')])
            num_col, num_row = self.crop_recon(eval_paths[0])[1:]
            eval_loader = data.DataLoader(ReconSet(window_imgs), batch_size=self.batch_size,
                                          shuffle=False, drop_last=False, num_workers=self.num_cpu)
            return eval_loader, num_col, num_row

        elif self.eval_type == 'Crops':
            #Just for looking at individual window performance
            window_imgs = np.concatenate([self.crop_recon(eval_paths[i])[0]
                                          for i in tqdm(range(len(eval_paths)), desc='Crop Images')])
            eval_loader = data.DataLoader(ReconSet(window_imgs), batch_size=1,
                                          shuffle=False, drop_last=False, num_workers=self.num_cpu)
            return eval_loader, None, None

        elif self.eval_type == 'Scaled':
            a, b, c = np.array(cv2.imread(eval_paths[0], 1)).shape
            alpha, beta = int(0.15 * a), int(0.15 * b)
            h = 1024
            w = 1024
            scale_dim = (w, h)
            scaled_imgs = np.concatenate([np.array(cv2.resize(cv2.imread(eval_paths[i], 1),
                                            scale_dim, interpolation=cv2.INTER_NEAREST)).reshape(1, h, w, c) / 255.0
                                            for i in range(len(eval_paths))])
            scaled_imgs = np.transpose(scaled_imgs, axes=(0, 3, 1, 2))
            print(scaled_imgs.shape)
            eval_loader = data.DataLoader(ReconSet(scaled_imgs), batch_size=1, #smaller batch size because bigger than normal runs
                                          shuffle=False, drop_last=False, num_workers=self.num_cpu)
            return eval_loader, None, None

        else:
            print('Wrong eval type, try again.')
            return None


