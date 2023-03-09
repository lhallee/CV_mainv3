import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Optimizer
from collections import defaultdict


def calculate_overlap_metrics(pred, gt):
    #Calculates various metrics for image segmentation models
    eps = 1e-6
    output = pred.clone().detach().flatten()
    target = gt.clone().detach().flatten()
    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    F1 = 2 * recall * precision / (recall + precision + 1e-6)
    return pixel_acc, dice, precision, recall, specificity, F1


#Losses
#https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class MultiNoiseLoss(nn.Module):
    def __init__(self, n_losses):
        """
        Initialise the module, and the scalar "noise" parameters (sigmas in arxiv.org/abs/1705.07115).
        If using CUDA, requires manually setting them on the device, even if the model is already set to device.
        """
        super(MultiNoiseLoss, self).__init__()
        self.n_losses = n_losses
        if torch.cuda.is_available():
            self.noise_params = torch.rand(n_losses, requires_grad=True, device="cuda:0")
        else:
            self.noise_params = torch.rand(n_losses, requires_grad=True)

    def DiceBCELoss(self, inputs, targets, smooth=1):
        inputs = inputs.flatten()
        targets = targets.flatten()
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets)
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

    def forward(self, SR, GT):
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)
        """

        losses = [self.DiceBCELoss(SR[:, :, :, i], GT[:, :, :, i]) for i in range(self.n_losses)]

        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += (1 / torch.square(self.noise_params[i])) * loss + torch.log(self.noise_params[i])

        return total_loss


class Dice_IOU_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice_IOU_Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IOU_loss = 1 - (intersection + smooth) / (union + smooth)
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        Dice_IOU = IOU_loss + dice_loss

        return Dice_IOU


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss

#Schedulers
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
