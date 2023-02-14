#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_processing import training_processing, eval_processing
from run_model import Solver
from torch.backends import cudnn
from evaluation import eval_solver


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
            plt.imshow(GTs[i][:,:,0], cmap='gray')
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

def main(config):
    #Takes argpase config settings runs the model with them
    config.output_ch = config.num_class
    cudnn.benchmark = True

    print(config)
    if config.mode == 'CV':
        from cross_validation import cross_validator
        solver = cross_validator(config)
        solver.run()

    if config.mode == 'eval':
        #eval mode is for evaluating the 3D reconstruction capabilities of a model
        data_setup = eval_processing(config)
        eval_loader, num_col, num_row = data_setup.eval_dataloader()
        #dataloader of eval data, number of columns in window split, number of rows in window split
        solver = eval_solver(config, eval_loader, num_col, num_row)
        solver.eval()

    #Can choose between real data in a path or generated data of squares of various sizes
    if config.data_type == 'Real':
        data_setup = training_processing(config)
        train_loader, valid_loader = data_setup.to_dataloader()
        print(len(train_loader), len(valid_loader))
        #vis_imgs, vis_GTs = train_loader.dataset[:25]
        #preview_crops(vis_imgs, vis_GTs, config.num_class)
        solver = Solver(config, train_loader, valid_loader)

    #Train utilizes random weights to train until stopping criteria of the number of epochs
    #then calls the test function
    if config.mode == 'train':
        solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--t', type=int, default=3, help='Number of recurrent steps')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for segmentation')
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--scheduler', type=str, default='cosine', help='None, exp, cosine')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/Vis/BigR2AttU_Net')

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--loss', type=str, default='DiceBCE', help='BCE, DiceBCE, IOU, CE, DiceIOU')
    parser.add_argument('--multi_loss', type=bool, default=False, help='Separate losses for each class or not')

    # Paths
    parser.add_argument('--model_path', type=str, default='None', help='Path for model weights')
    parser.add_argument('--train_img_path', type=str, default='./img_data/train_img/',
                        help='Path for training images')
    parser.add_argument('--val_img_path', type=str, default='./img_data/val_img/',
                        help='Path for validation images')
    parser.add_argument('--val_GT_paths', type=list, default=['./img_data/val_GT_1/', './img_data/val_GT_2/'],
                        help='Path for validation GT')
    parser.add_argument('--GT_paths', type=list, default=['./img_data/GT_1/', './img_data/GT_2/'],
                        help='List of paths for training GT (one for each class)')
    parser.add_argument('--eval_img_path', type=str, default='./img_data/eval_img/',
                        help='Images for 2D reconstruction evaluation')
    parser.add_argument('--result_path', type=str, default='./result/', help='Path for results')

    # misc
    parser.add_argument('--mode', type=str, default='train', help='train, eval, CV')
    parser.add_argument('--cuda_idx', type=int, default=0, help='Cuda index')
    parser.add_argument('--data_type', type=str, default='Real', help='Real or Mock data')
    parser.add_argument('--eval_type', type=str, default='Windowed', help='Type of evaluation. Windowed, Crops, Scaled')
    parser.add_argument('--stop', type=float, default=0.975, help='Minimum stopping criteria for unet score')

    config = parser.parse_args()
    main(config)
