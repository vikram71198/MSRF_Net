import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchgeometry.losses.dice import DiceLoss
from torchvision.models.resnet import BasicBlock

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss   = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_1, pred_2, pred_3, pred_canny, msk, canny_label):
        loss_pred_1 = self.ce_loss(pred_1, msk) + self.dice_loss(pred_1, msk)
        loss_pred_2 = self.ce_loss(pred_2, msk) + self.dice_loss(pred_2, msk)
        loss_pred_3 = self.ce_loss(pred_3, msk) + self.dice_loss(pred_3, msk)
        loss_canny = self.bce_loss(pred_canny, canny_label)
        loss = loss_pred_3 + loss_pred_1 + loss_pred_2 + loss_canny

        return loss