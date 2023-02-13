import numpy as np
import torch
import scipy.interpolate
from skimage import filters
import cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from matplotlib import pyplot as plt
from datetime import datetime


class eval_solver:
    def __init__(self, config, eval_loader, num_col=0, num_row=0):
        self.eval_loader = eval_loader
        self.eval_type = config.eval_type
        self.eval_path = config.eval_type
        self.num_col = num_col
        self.num_row = num_row
        self.model_type = config.model_type
        self.t = config.t
        self.unet = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.batch_size = config.batch_size
        self.dim = config.image_size
        self.result_path = config.result_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        recon = np.zeros(((self.num_col + 1) * self.dim, (self.num_row + 1) * self.dim))
        k = 0
        halfdim = int(self.dim/2)
        for i in range(self.num_col):
            for j in range(self.num_row):
                inner = SR[k][int(1.5 * halfdim):int(2.5 * halfdim), int(1.5 * halfdim):int(2.5 * halfdim)]
                recon[i * self.dim:(i + 1) * self.dim, j * self.dim:(j + 1) * self.dim] = inner
                k += 1

        '''
        W, H = recon.shape
        x_col, y_col = np.array(range(W)), np.array(range(H))
        x_high, y_high, = np.arange(0, W, super_ratio), np.arange(0, H, super_ratio)
        filt_img = filters.threshold_local(recon, filter_radius)
        filt_img[filt_img < thresh_ratio] = 0.0
        filt_img[filt_img >= thresh_ratio] = 1.0
        filt_set_func = scipy.interpolate.RectBivariateSpline(x_col, y_col, filt_img)
        filt_func_img = filt_set_func(x_high, y_high)
        return filt_func_img
        '''
        return recon

    @torch.no_grad()  # don't update weights while evaluating
    def eval(self):
        now = str(datetime.now())[11:] #just time
        self.build_model()  # rebuild model
        try:
            self.unet.load_state_dict(torch.load(self.model_path, map_location=self.device))  # load pretrained weights
        except:
            input_path = input('Please type the path to the desired saved weights: ')
            self.unet.load_state_dict(torch.load(input_path, map_location=self.device))  # load pretrained weights
        loop = tqdm(self.eval_loader, leave=True)
        SRs = np.concatenate([self.unet(batch.to(self.device)).detach().cpu().numpy() for batch in loop])
        SRs = np.transpose(SRs, axes=(0, 2, 3, 1))  # back to normal img format
        if self.eval_type == 'Crops':
            for i in tqdm(range(len(SRs))):
                recon = np.hstack((SRs[i][:,:,1], SRs[i][:,:,0]))  # stack results together
                plt.imsave(self.result_path + 'eval' + now + self.eval_type + str(i) + '_img.png', recon)

        elif self.eval_type == 'Windowed':
            for i in tqdm(range(int(len(SRs)/(self.num_row * self.num_col))), desc='Evaluation'):
                try:
                    single_SR = SRs[i * self.num_row * self.num_col:(i+1) * self.num_row * self.num_col]
                    recon_lob = self.window_recon(single_SR[:, :, :, 0])
                    recon_hev = self.window_recon(single_SR[:, :, :, 1])
                    #recon_lob = filters.threshold_local(recon_lob, 101)
                    #recon_hev = filters.threshold_local(recon_hev, 21)
                    recon_lob = recon_lob.reshape(recon_lob.shape[0], recon_lob.shape[1]) #reshape from dim,dim,1 to dim, dim
                    recon_hev = recon_hev.reshape(recon_hev.shape[0], recon_hev.shape[1])
                    recon = np.hstack((recon_hev, recon_lob)) #stack results together
                    plt.imsave(self.result_path + 'eval' + now + str(i) + '.png', recon)
                except:
                    continue
