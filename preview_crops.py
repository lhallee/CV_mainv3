import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class ImageSet(Dataset):
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


class ValidSet(Dataset):
    # Custom pytorch dataset, simply indexes imgs
    def __init__(self, imgs):
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = torch.tensor(self.imgs[index], dtype=torch.float)
        return img


def preview_crops(imgs, GTs, num_class=2):
    #Displays training crops from dataloaders
    rows = 1
    columns = num_class + 1
    #Back to normal image format
    imgs = np.transpose(np.array(imgs), axes=(0, 2, 3, 1))
    GTs = np.transpose(np.array(GTs), axes=(0, 2, 3, 1))
    for i in range(len(imgs)):
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title('Img')
        if num_class == 1:
            fig.add_subplot(rows, columns, 2)
            plt.imshow(GTs[i][:, :, 0], cmap='gray')
            plt.axis('off')
            plt.title('GT')
            plt.show()
        else:
            fig.add_subplot(rows, columns, 2)
            plt.imshow(GTs[i][:, :, 0], cmap='gray')
            plt.axis('off')
            plt.title('GT')
            fig.add_subplot(rows, columns, 3)
            plt.imshow(GTs[i][:, :, 1], cmap='gray')
            plt.axis('off')
            plt.title('GT')
            plt.show()

train_img_path = 'C:/Users/Logan Hallee/Desktop/LN IMAGES/training_data/256mini_img.npy'
train_GT_path = 'C:/Users/Logan Hallee/Desktop/LN IMAGES/training_data/256mini_GT.npy'
valid_img_path = 'C:/Users/Logan Hallee/Desktop/LN IMAGES/training_data/256val_img_data.npy'
val_GT_1 = ''
val_GT_2 = ''

# Dataloaders
print('Loading Data')
train_img_data = np.load(train_img_path, allow_pickle=True)
train_GT_data = np.load(train_GT_path, allow_pickle=True)
valid_img_data = np.load(valid_img_path, allow_pickle=True)
train_ds = ImageSet(train_img_data, train_GT_data)
valid_ds = ValidSet(valid_img_data)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)
print('Dataloaders compiled')


num = 100
imgs, GTs = train_loader.dataset[:num]
preview_crops(imgs, GTs)



