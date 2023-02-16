#!/usr/bin/env python3
import argparse
import torch
from torch.utils import data
from trainer import Trainer
from torch.backends import cudnn
#from evaluation import eval_solver

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


def main(config):
    #Takes argpase config settings runs the model with them
    cudnn.benchmark = True

    #print(config)
    if config.mode == 'CV':
        from cross_validation import cross_validator
        solver = cross_validator(config)
        solver.run()

    #Train utilizes random weights to train until stopping criteria of the number of epochs
    #then calls the test function
    if config.mode == 'train':
        train_loader = torch.load(config.train_loader_path)
        valid_loader = torch.load(config.valid_loader_path)
        print(len(train_loader), len(valid_loader))
        solver = Trainer(config, train_loader, valid_loader)
        solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=768)
    parser.add_argument('--t', type=int, default=3, help='Number of recurrent steps')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for segmentation')
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--scheduler', type=str, default='cosine', help='None, exp, cosine')
    parser.add_argument('--model_type', type=str, default='TestNet', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/Vis/BigR2AttU_Net')

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--loss', type=str, default='DiceBCE', help='BCE, DiceBCE, IOU, CE, DiceIOU')
    parser.add_argument('--multi_loss', type=bool, default=False, help='Separate losses for each class or not')

    # Paths
    parser.add_argument('--model_path', type=str, default='None', help='Path for model weights')
    parser.add_argument('--result_path', type=str, default='./result/', help='Path for results')
    parser.add_argument('--train_loader_path', type=str, default='./dataloaders/train_dataloader.pth')
    parser.add_argument('--valid_loader_path', type=str, default='./dataloaders/valid_dataloader.pth')
    parser.add_argument('--val_GT_paths', type=list, default=['./img_data/val_GT_1/', './img_data/val_GT_2/'],
                        help='Path for validation GT')

    # misc
    parser.add_argument('--mode', type=str, default='train', help='train, eval, CV')
    parser.add_argument('--cuda_idx', type=int, default=0, help='Cuda index')
    parser.add_argument('--data_type', type=str, default='Real', help='Real or Mock data')
    parser.add_argument('--eval_type', type=str, default='Windowed', help='Type of evaluation. Windowed, Crops, Scaled')
    parser.add_argument('--stop', type=float, default=0.975, help='Minimum stopping criteria for unet score')

    config = parser.parse_args()
    main(config)
