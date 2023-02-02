import torch

def _calculate_overlap_metrics(pred, gt):
    #Calculates various metrics for image segmentation models
    eps = 1e-6
    output = pred.view(-1, )
    target = gt.view(-1, ).float()
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