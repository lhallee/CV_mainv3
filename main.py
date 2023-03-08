#!/usr/bin/env python3
import argparse
from trainer import Trainer
#from monai_trainer_multi import Trainer as monaimultiTrainer
from torch.backends import cudnn
#from evaluation import eval_solver

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
        if config.multi:
            pass
            #solver = monaimultiTrainer(config)
        else:
            solver = Trainer(config)
        solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='Number of recurrent steps')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for segmentation')
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--scheduler', type=str, default='cyclic', help='None, exp, cosine, cyclic')
    parser.add_argument('--model_type', type=str, default='R2AttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/Vis/BigNeT')

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--loss', type=str, default='DiceBCE', help='BCE, DiceBCE, IOU, CE, DiceIOU')

    # Paths
    parser.add_argument('--model_path', type=str, default='None', help='Path for model weights')
    parser.add_argument('--result_path', type=str, default='./result/', help='Path for results')
    parser.add_argument('--train_img_path', type=str, default='./tiny_data/train_img_data.npy')
    parser.add_argument('--train_GT_path', type=str, default='./tiny_data/train_GT_data.npy')
    parser.add_argument('--val_img_path', type=str, default='./tiny_data/val_img_data.npy')
    #parser.add_argument('--val_GT_paths', type=list, default=['./val_GT_1/', './val_GT_2/'])
    parser.add_argument('-ap', '--val_GT_paths', action='append', help='<Required> Set flag', required=True)

    # misc
    parser.add_argument('--mode', type=str, default='train', help='train, eval, CV')
    parser.add_argument('--cuda_idx', type=int, default=0, help='Cuda index')
    parser.add_argument('--data_type', type=str, default='Real', help='Real or Mock data')
    parser.add_argument('--eval_type', type=str, default='Windowed', help='Type of evaluation. Windowed, Crops, Scaled')
    parser.add_argument('--stop', type=float, default=0.975, help='Minimum stopping criteria for unet score')
    parser.add_argument('--use_viewer', type=bool, default=True)
    parser.add_argument('--multi', type=bool, default=False, help='If multi gpu trainer or not')
    parser.add_argument('--num_workers', type=int, default=4)

    config = parser.parse_args()
    main(config)
