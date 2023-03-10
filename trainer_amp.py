import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from skimage.util import view_as_windows
from models import *
from model_parts import CosineWarmupScheduler, _calculate_overlap_metrics, DiceBCELoss


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


class Trainer(object):
    def __init__(self, config):
        #Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.best_unet_score = 0
        self.stop = config.stop

        #Dataloaders
        print('Loading Data')
        train_img_data = np.load(config.train_img_path, allow_pickle=True)
        train_GT_data = np.load(config.train_GT_path, allow_pickle=True)
        valid_img_data = np.load(config.val_img_path, allow_pickle=True)
        train_ds = ImageSet(train_img_data, train_GT_data)
        valid_ds = ValidSet(valid_img_data)
        self.train_loader = DataLoader(train_ds, num_workers=config.num_workers,
                                             batch_size=config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_ds, num_workers=config.num_workers,
                                             batch_size=config.batch_size, shuffle=False)
        print('Dataloaders compiled')

        #Model
        self.model_type = config.model_type
        self.unet = None
        self.best_unet = None #might not need
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
        self.t = config.t

        #Paths
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.val_GT_paths = config.val_GT_paths
        self.save_path = None

        #MISC
        self.num_col = None
        self.num_row = None
        self.use_viewer = config.use_viewer


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
        elif self.model_type == 'TestNet':
            self.unet = TestNet(img_ch=self.img_ch, output_ch=self.num_class)
            self.valid(0)

        if self.loss == 'DiceBCE':
            self.criterion = DiceBCELoss().to(self.device)

        self.optimizer = torch.optim.AdamW(list(self.unet.parameters()), self.lr)
        if self.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1)
        elif self.scheduler == 'cosine':
            self.scheduler = CosineWarmupScheduler(self.optimizer, warmup=len(self.train_loader) * 4,
                                                   max_iters=len(self.train_loader) * self.num_epochs)
        elif self.scheduler == 'cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, self.lr, 5 * self.lr,
                                                               mode='Exp_range', cycle_momentum=False)

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

    def window_recon(self, SR):
        qdim = int(self.dim / 4)
        hdim = int(self.dim / 2)
        recon = np.zeros(((self.num_col + 1) * hdim, (self.num_row + 1) * hdim))
        k = 0
        for i in range(self.num_col):
            for j in range(self.num_row):
                inner = SR[k][qdim:3*qdim, qdim:3*qdim]
                recon[i * hdim:(i + 1) * hdim, j * hdim:(j + 1) * hdim] = inner
                k += 1
        return recon.reshape((self.num_col + 1) * hdim, (self.num_row + 1) * hdim, 1)

    def viewer(self, pred, GT, val):
        recons = np.vstack([pred[:, :, i] for i in range(self.num_class)])
        GTs = np.vstack([GT[:, :, i] for i in range(self.num_class)])
        plot = np.hstack((recons, GTs))
        if val:
            ratio = 0.05
            plot = np.array(cv2.resize(plot, (int(plot.shape[1] * ratio), int(plot.shape[0] * ratio))))
        #cv2.imwrite(self.result_path + str(epoch) + 'valimg.png', plot)
        plt.imshow(plot)
        #plt.imsave(plot)
        plt.show()

    def train(self):
        self.build_model()
        scaler = torch.cuda.amp.GradScaler()
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
                self.unet.train()
                images = images.to(self.device)
                GT = GT.to(self.device)
                with torch.cuda.amp.autocast():
                    SR = self.unet(images) #SR : Segmentation Result
                    loss = self.criterion(SR, GT)
                epoch_loss += loss.item()
                #Backprop + optimize
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                scaler.update()
                if length % 100 == 0 and self.use_viewer:
                    self.viewer(np.transpose(SR.detach().cpu(), axes=(0, 2, 3, 1))[0],
                                np.transpose(GT.detach().cpu(), axes=(0, 2, 3, 1))[0],
                                False)
                # calculate metrics
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

    @torch.no_grad() #don't update weights during validation
    def valid(self, epoch):
        self.unet.eval()
        GTs = []
        for i in range(self.num_class):
            GTs.append(natsorted(glob(self.val_GT_paths[i] + '*'))[::-1])
            # the [::-1] is strictly unnecessary!!!!!!!!!!!! ONLY BECAUSE OF THE CURRENT 256 VAL SET
        if self.num_col is None:
            a, b = np.array(cv2.imread(GTs[0][0], 2)).shape
            self.val_GT = np.concatenate([
                np.concatenate([np.array(cv2.imread(GTs[i][j], 2), dtype=np.float32).reshape(1, a, b, 1)
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

        loop = tqdm(self.valid_loader, desc='Validation', leave=True)
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                SRs = np.concatenate([self.unet(batch.to(self.device)).detach().cpu().numpy() for batch in loop])
        else:
            SRs = np.concatenate([self.unet(batch.to(self.device)).detach().cpu().numpy() for batch in loop])
        SRs = np.transpose(SRs, axes=(0, 2, 3, 1))  # back to normal img format
        a, b, c = self.val_GT[0].shape
        for i in tqdm(range(int(len(SRs) / (self.num_row * self.num_col))), desc='Reconstruction'):
            single_SR = SRs[i * self.num_row * self.num_col:(i + 1) * self.num_row * self.num_col]
            recon = np.concatenate([self.window_recon(single_SR[:, :, :, i])
                                    for i in range(self.num_class)], axis=2)
            GT = self.val_GT[i][:int(int(a / (self.dim / 2)) * self.dim / 2),
                 :int(int(b / (self.dim / 2)) * self.dim / 2)]
            if self.use_viewer:
                self.viewer(recon, GT, True)
            # Calculate metrics
            _acc, _DC, _PC, _RE, _SP, _F1 = _calculate_overlap_metrics(torch.tensor(recon), torch.tensor(GT))
            acc += _acc.item()
            DC += _DC.item()
            RE += _RE.item()
            SP += _SP.item()
            PC += _PC.item()
            F1 += _F1.item()
            length += 1

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
        #Only saved when validation metrics are improved on average
        if unet_score > self.best_unet_score:
            self.best_unet = self.unet.state_dict()
            self.best_unet_score = unet_score
            self.best_epoch = epoch
            if self.model_path != 'None':
                self.save_path = self.model_path + self.model_type + self.data_type + self.loss \
                                 + str(self.best_epoch) + str(self.lr) + '.pkl'  # descriptive name from settings
            else:
                self.save_path = self.model_type + self.data_type + self.loss + str(self.best_epoch) \
                                 + str(self.lr) + '.pkl'  # descriptive name from settings
            print('Best %s model score : %.4f' % (self.model_type, self.best_unet_score))
            torch.save(self.best_unet, self.save_path)




