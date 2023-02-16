import numpy as np
import os
import cv2
import skorch
import torch
from skimage.util import view_as_windows
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from model_parts import *


from models import *
from data_processing import *


# custom optimizer to encapsulate Adam
def make_lookahead(parameters, optimizer_cls, k, alpha, **kwargs):
    optimizer = optimizer_cls(parameters, **kwargs)
    return Lookahead(optimizer=optimizer, k=k, alpha=alpha)


class cross_validator:
    def __init__(self, config):
        self.img_path = config.img_path
        self.GT_path = config.GT_path
        self.dim = config.image_size
        self.num_class = config.num_class
        self.train_per = config.train_per
        self.batch_size = config.batch_size
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.num_cpu = os.cpu_count()
        self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def crop_augment(self, img, GT=None, hev=None, lob=None):
        if self.num_class > 1:
            # if multiclass, put lob and hev ground truths together
            img = np.array(cv2.imread(img, 1)) / 255.0  # read and scale img
            hev = np.array(cv2.imread(hev, 2), dtype=np.float32).reshape(img.shape[0], img.shape[1], 1)
            lob = np.array(cv2.imread(lob, 2), dtype=np.float32).reshape(img.shape[0], img.shape[1], 1)
            GT = np.concatenate((lob, hev), axis=2)
        else:
            img = np.array(cv2.imread(img, 1)) / 255.0  # read and scale img
            GT = np.array(cv2.imread(GT, 2), dtype=np.float32).reshape(img.shape[0], img.shape[1], 1)

        imgs = view_as_windows(img, (self.dim, self.dim, 3), step=self.dim)
        GTs = view_as_windows(GT, (self.dim, self.dim, self.num_class), step=self.dim)
        a, b, c, d, e, f = imgs.shape
        imgs = imgs.reshape(a * b, self.dim, self.dim, 3)  # reshape windowed output into num_images, dim, dim channel
        GTs = GTs.reshape(a * b, self.dim, self.dim, self.num_class)

        GTs[GTs < 1] = 0  # GT pixels should have 0 background
        GTs[GTs > 0] = 1  # and actual features 1
        return imgs, GTs

    def to_dataloader_multi(self):
        img_paths = natsorted(glob(self.img_path + '*.png'))[:1]  # natural sort
        hev_paths = natsorted(glob(self.GT_path + '*hev*'))[:1]
        lob_paths = natsorted(glob(self.GT_path + '*Lob*'))[:1]
        assert len(img_paths) * self.num_class == len(hev_paths) + len(lob_paths), 'Need GT for every Image.'
        # Combine results from each image path into one array
        crop_imgs = np.concatenate([self.crop_augment(img=img_paths[i], hev=hev_paths[i], lob=lob_paths[i])[0]
                                    for i in tqdm(range(len(img_paths)))], axis=0)
        crop_GTs = np.concatenate([self.crop_augment(img=img_paths[i], hev=hev_paths[i], lob=lob_paths[i])[1]
                                   for i in tqdm(range(len(img_paths)))], axis=0)
        # numpy array to torch tensor, move around columns for pytorch convolution
        crop_imgs = torch.tensor(np.transpose(crop_imgs, axes=(0, 3, 1, 2)), dtype=torch.float32)
        crop_GTs = torch.tensor(np.transpose(crop_GTs, axes=(0, 3, 1, 2)), dtype=torch.float32)
        return crop_imgs, crop_GTs


    def run(self):
        X, y = self.to_dataloader_multi()

        net = skorch.NeuralNet(
            module=self.unet,
            max_epochs=10,
            criterion=DiceBCELoss(),
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True,
            callbacks=[skorch.callbacks.ProgressBar()]
        )
        net.set_params(train_split=False, verbose=0)
        params = {
            'module__t': [3],
            'optimizer__weight_decay': [0.001],
            'optimizer__lr': [0.001]
        }
        gs = GridSearchCV(net, params, cv=2, scoring='accuracy', verbose=2)
        gs.fit(X, y)
        print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
