import os
import cv2
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from metrics import _calculate_overlap_metrics
from models import *
from custom_model_components import *
from natsort import natsorted
from glob import glob
from skimage.util import view_as_windows



class Solver(object):
    def __init__(self, config, train_loader, valid_loader):
        #Dataloaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        #Model
        self.model_type = config.model_type
        self.t = config.t
        self.unet = None
        self.best_unet = None #might not need
        self.unet_path = None #save path?
        self.optimizer = None
        self.best_epoch = None #might not need
        self.criterion = None
        self.loss = config.loss
        self.img_ch = config.img_ch
        self.num_class = config.num_class
        self.data_type = config.data_type
        self.scheduler = config.scheduler
        self.dim = config.image_size

        #Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        #Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.best_unet_score = 0
        self.stop = config.stop
        self.multi_losses = config.multi_loss

        #Paths
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.val_GT_paths = config.val_GT_paths

        #MISC
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model_path != 'None':
            self.unet_path = self.model_path + self.model_type + self.data_type + self.loss\
                             + str(self.num_epochs) + str(self.lr) + '.pkl'  # descriptive name from settings
        else:
            self.unet_path = self.model_type + self.data_type + self.loss + str(self.num_epochs)\
                             + str(self.lr) + '.pkl' #descriptive name from settings
        self.num_col = None
        self.num_row = None


    def build_model(self):
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.num_class)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch, output_ch=self.num_class, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.num_class)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.num_class, t=self.t)
        elif self.model_type == 'BigNET':
            self.unet = BigR2AttU_Net(img_ch=self.img_ch, output_ch=self.num_class, t=self.t)
        elif self.model_type == 'Vis':
            self.unet = R2AttU_Net_vis(img_ch=self.img_ch, output_ch=self.num_class, t=self.t)


        if self.loss == 'BCE':
            self.criterion = nn.BCELoss()
        elif self.loss == 'DiceBCE':
            self.criterion = DiceBCELoss()
        elif self.loss == 'IOU':
            self.criterion = IoULoss()
        elif self.loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss == 'DiceIOU':
            self.criterion = Dice_IOU_Loss()

        self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, (self.beta1, self.beta2))
        if self.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1)
        elif self.scheduler == 'cosine':
            self.scheduler = CosineWarmupScheduler(self.optimizer, warmup=len(self.train_loader) * 4,
                                                   max_iters=len(self.train_loader) * self.num_epochs)
        self.unet.to(self.device)
        #self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def train(self):
        self.build_model()
        if self.model_path != 'None':
            self.unet.load_state_dict(torch.load(self.model_path, map_location=self.device))
        epoch = 0
        #Unet score is the average of our metrics calculated from validation
        while epoch < self.num_epochs and self.best_unet_score < self.stop:
            epoch_loss = 0
            acc = 0.  # Accuracy
            RE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            DC = 0.  # Dice Coefficient
            length = 0
            pbar_train = tqdm(total=len(self.train_loader), desc='Training')
            for images, GT in self.train_loader:
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = self.unet(images) #SR : Segmentation Result
                if self.multi_losses:
                    loss_hev = self.criterion(SR[:,:,1], GT[:,:,1])
                    loss_lob = self.criterion(SR[:,:,0], GT[:,:,0])
                    smooth = 1
                    if loss_lob > loss_hev:
                        ratio = (loss_lob + smooth) / (loss_hev + smooth)
                    else:
                        ratio = (loss_hev + smooth) / (loss_lob + smooth)
                    loss = (loss_hev + loss_lob) * ratio / 2
                else:
                    loss = self.criterion(SR, GT)
                epoch_loss += loss.item()
                #Backprop + optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                #calculate metrics
                _acc, _DC, _PC, _RE, _SP, _F1 = _calculate_overlap_metrics(SR.detach().cpu(), GT.detach().cpu())
                acc += _acc.item()
                DC += _DC.item()
                RE += _RE.item()
                SP += _SP.item()
                PC += _PC.item()
                F1 += _F1.item()
                length += 1 #for epoch metrics and number of batches
                pbar_train.update(1) #progress bar update
            acc = acc / length
            RE = RE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            DC = DC / length
            # Print the log info
            print(
                'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, RE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, DC: %.4f' % (
                    epoch + 1, self.num_epochs, epoch_loss, acc, RE, SP, PC, F1, DC))
            pbar_train.close() #close progress bar
            self.valid(epoch) #call validation
            epoch += 1

    def window_recon(self, SR):
        recon = np.zeros(((self.num_col + 1) * self.dim, (self.num_row + 1) * self.dim))
        k = 0
        qdim = int(self.dim / 4)
        hdim = int(self.dim / 2)
        for i in range(self.num_col):
            for j in range(self.num_row):
                inner = SR[k][qdim:self.dim, qdim:self.dim]
                recon[i * hdim:(i + 1) * hdim, j * hdim:(j + 1) * hdim] = inner
                k += 1
        return recon


    @torch.no_grad() #don't update weights during validation
    def valid(self, epoch):
        GTs = []
        for i in range(self.num_class):
            GTs.append(natsorted(glob(self.val_GT_paths[i] + '*'))[:1])
        if self.num_col is None:
            self.val_GT = np.concatenate([
                np.concatenate([np.array(cv2.imread(GTs[i][j], 2), dtype=np.float32).reshape(1,
                                                                                             np.array(
                                                                                                 cv2.imread(GTs[i][j],
                                                                                                            2)).shape[
                                                                                                 0],
                                                                                             np.array(
                                                                                                 cv2.imread(GTs[i][j],
                                                                                                            2)).shape[
                                                                                                 1], 1)
                                for j in range(len(GTs[i]))], axis=0)
                for i in range(len(GTs))], axis=3)
            a, b, c, d = view_as_windows(np.array(cv2.imread(GTs[0][0], 2)), (self.dim, self.dim),
                                               step=int(self.dim / 2)).shape
            self.num_col, self.num_row = a, b

        acc = 0.  # Accuracy
        RE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        DC = 0.  # Dice Coefficient
        length = 0

        loop = tqdm(self.valid_loader, leave=True)
        SRs = np.concatenate([self.unet(batch.to(self.device)).detach().cpu().numpy() for batch in loop])
        SRs = np.transpose(SRs, axes=(0, 2, 3, 1))  # back to normal img format

        for i in tqdm(range(int(len(SRs) / (self.num_row * self.num_col))), desc='Evaluation'):
            try:
                single_SR = SRs[i * self.num_row * self.num_col:(i + 1) * self.num_row * self.num_col]
                recon = np.concatenate([self.window_recon(single_SR[:, :, :, i])
                                        for i in range(len(self.val_GT_paths))], axis=3)
                print(recon.shape)
                GT = self.val_GT[i]
                print(GT.shape)
                # Calculate metrics
                _acc, _DC, _PC, _RE, _SP, _F1 = _calculate_overlap_metrics(recon, GT)
                acc += _acc.item()
                DC += _DC.item()
                RE += _RE.item()
                SP += _SP.item()
                PC += _PC.item()
                F1 += _F1.item()
                length += 1
            except:
                continue

        acc = acc / length
        RE = RE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        DC = DC / length
        #unet_score is an average of metrics
        unet_score = (F1 + DC + PC + SP + RE + acc) / 6
        print('[Validation] Acc: %.4f, RE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, DC: %.4f' % (
            acc, RE, SP, PC, F1, DC))
        #Save Best U-Net model
        if unet_score > self.best_unet_score:
            self.best_unet_score = unet_score
            self.best_epoch = epoch
            self.best_unet = self.unet.state_dict()
            print('Best %s model score : %.4f' % (self.model_type, self.best_unet_score))
            torch.save(self.best_unet, self.unet_path)
            #only saved when validation metrics are improved on average




