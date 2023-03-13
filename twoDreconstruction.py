import numpy as np
import torch
import cv2
import os
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.util import view_as_windows
from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from matplotlib import pyplot as plt


class ReconSet(Dataset):
    # Custom pytorch dataset, simply indexes imgs
    def __init__(self, imgs):
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = torch.tensor(self.imgs[index], dtype=torch.float)
        return img


class eval_solver:
    def __init__(self, config):
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataloader
        print('Loading Data')
        self.eval_section_paths = config.eval_img_paths
        eval_paths = []
        self.indexes = []
        for i in range(len(self.eval_section_paths)):
            eval_paths = eval_paths + glob(self.eval_section_paths[i] + '*')
            self.indexes.append(len(self.eval_section_paths[i]))
        eval_paths = natsorted(eval_paths)
        eval_imgs, self.num_row, self.num_col = self.eval_dataarray(eval_paths)
        eval_ds = ReconSet(eval_imgs)
        self.eval_loader = DataLoader(eval_ds, num_workers=config.num_workers,
                                       batch_size=config.batch_size, shuffle=False)
        print('Dataloaders compiled')

        # Settings
        self.eval_type = config.eval_type
        self.model_type = config.model_type
        self.t = config.t
        self.unet = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.batch_size = config.batch_size
        self.dim = config.image_size


        # Paths
        self.model_path = config.model_path

        # Create section paths
        self.result_paths = []
        for i in range(len(self.eval_section_paths)):
            for j in range(len(self.output_ch)):
                path = self.eval_section_paths[i] + str(j) + '_section_' + str(i + 1) + '/'
                if not os.path.exists(path):
                    os.mkdir(path)
                self.result_paths.append(path)

    def crop_eval(self, img):
        img = np.array(cv2.imread(img, 1)) / 255.0  # load and scale img
        imgs = view_as_windows(img, (self.dim, self.dim, 3), step=int(self.dim / 2))
        a, b, c, d, e, f = imgs.shape
        imgs = imgs.reshape(a * b, self.dim, self.dim, 3)
        return imgs, a, b


    def eval_dataarray(self, eval_paths):
        eval_imgs = np.concatenate([self.crop_eval(img=eval_paths[i])[0]
                                   for i in tqdm(range(len(eval_paths)), desc='Cropping Eval')], axis=0)
        num_row, num_col = self.crop_eval(img=eval_paths[0])[1:]
        print(eval_imgs.shape)
        #numpy array to torch tensor, move around columns for pytorch convolution
        eval_imgs = np.transpose(eval_imgs, axes=(0, 3, 1, 2))

        return eval_imgs, num_row, num_col


    def build_model(self):
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        self.unet.to(self.device)

    def window_recon(self, SR):
        qdim = int(self.dim / 4)
        hdim = int(self.dim / 2)
        recon = np.zeros(((self.num_col + 1) * hdim, (self.num_row + 1) * hdim))
        k = 0
        for i in range(self.num_col):
            for j in range(self.num_row):
                inner = SR[k][qdim:3 * qdim, qdim:3 * qdim]
                recon[i * hdim:(i + 1) * hdim, j * hdim:(j + 1) * hdim] = inner
                k += 1
        return recon.reshape((self.num_col + 1) * hdim, (self.num_row + 1) * hdim, 1)


    @torch.no_grad()  # don't update weights while evaluating
    def eval(self):
        self.build_model()  # build model
        try:
            self.unet.load_state_dict(torch.load(self.model_path, map_location=self.device))  # load pretrained weights
        except:
            input_path = input('Please type the path to the desired saved weights: ')
            self.unet.load_state_dict(torch.load(input_path, map_location=self.device))  # load pretrained weights
        self.unet.eval()
        loop = tqdm(self.eval_loader, leave=True)
        SRs = np.concatenate([self.unet(batch.to(self.device)).detach().cpu().numpy() for batch in loop])
        SRs = np.transpose(SRs, axes=(0, 2, 3, 1))  # back to normal img format

        if self.eval_type == 'Windowed':
            sec, k = 0, 0
            index = self.indexes[0]

            for i in tqdm(range(int(len(SRs) / (self.num_row * self.num_col))), desc='Reconstruction'):
                # Pull out individual slices from batched dataloader
                single_SR = SRs[i * self.num_row * self.num_col:(i + 1) * self.num_row * self.num_col]
                # Reconstruct in 2D one class at a time and stack together
                recon = np.concatenate([self.window_recon(single_SR[:, :, :, i])
                                        for i in range(self.output_ch)], axis=2)
                # Save each class in its section / class directory
                for j in range(len(self.output_ch)):
                    # First image is saved to result_paths sec, the next images are saved sec + j where there is one for each class
                    plt.imsave(self.result_paths[sec + j] + str(j) + '_eval_' + str(i) + '.png', recon[:, :, :, j])

                k += 1
                if k == index: # when the section has been saturated
                    sec += self.output_ch # move the section by the number of classes (so result_paths starts at the right index
                    index = self.indexes[int(sec / self.output_ch)] # get the new amount in the next section

