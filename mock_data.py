import cv2 as cv
import os
import torch
import numpy as np
from torch.utils import data
from skimage.util import view_as_windows
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def create_square(dim, x, y, z):
    RED = (0, 0, 1)
    Label = (1)
    pts = [(x - z, y - z), (x + z, y - z), (x + z, y + z), (x - z, y + z)]
    img = np.zeros((dim, dim, 3), np.uint8)
    GT = np.zeros((dim, dim, 1), np.uint8)
    img = cv.fillPoly(img, np.array([pts]), RED)
    GT = cv.fillPoly(GT, np.array([pts]), Label)
    img = img.reshape(1, dim, dim, 3)
    GT = GT.reshape(1, dim, dim, 1)
    return img, GT


class ImageSet(data.Dataset):
    #Indexes generated data and ground truths
    def __init__(self, imgs, GTs):
        self.imgs = imgs
        self.GTs = GTs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = self.imgs[index]
        GT = self.GTs[index]
        return img, GT


def crop_augment_mock(img, GT, dim, step):
    imgs = view_as_windows(img, (dim, dim, 3), step=step)
    GTs = view_as_windows(GT, (dim, dim, 1), step=step)
    a, b, c, d, e, f = imgs.shape
    imgs = imgs.reshape(a * b, dim, dim, 3)
    GTs = GTs.reshape(a * b, dim, dim, 1)
    GTs[GTs < 1] = 0
    GTs[GTs > 0] = 1
    return imgs, GTs


def to_dataloader_mock(dim=256, train_per=0.7,
                       batch_size=8, num_cpu=os.cpu_count()):
    imgs = np.concatenate([create_square(3000, 1500, 1500, i * 50)[0] for i in range(15)])
    GTs = np.concatenate([create_square(3000, 1500, 1500, i * 50)[1] for i in range(15)])
    print(imgs.shape, GTs.shape)
    assert len(imgs) == len(GTs), 'Need GT for every Image.'
    crop_imgs = np.concatenate([crop_augment_mock(imgs[i], GTs[i], dim, int(dim/2))[0] for i in tqdm(range(len(imgs)))])
    crop_GTs = np.concatenate([crop_augment_mock(imgs[i], GTs[i], dim, int(dim/2))[1] for i in tqdm(range(len(imgs)))])
    crop_imgs = torch.tensor(np.transpose(crop_imgs, axes=(0, 3, 1, 2)), dtype=torch.float)
    crop_GTs = torch.tensor(np.transpose(crop_GTs, axes=(0, 3, 1, 2)), dtype=torch.float)
    X_train, X_mem, y_train, y_mem = train_test_split(crop_imgs, crop_GTs, train_size=train_per)
    X_valid, X_test, y_valid, y_test = train_test_split(X_mem, y_mem, test_size=0.33)
    train_data = ImageSet(X_train, y_train)
    valid_data = ImageSet(X_valid, y_valid)
    test_data = ImageSet(X_test, y_test)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_cpu)
    val_loader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_cpu)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_cpu)
    return train_loader, val_loader, test_loader

