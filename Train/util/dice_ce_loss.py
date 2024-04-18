import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.6, weight_dice=0.4, ignore_index=None, label_smoothing=0.1):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.dice_loss = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, ignore_index=ignore_index, smooth=label_smoothing)

    def forward(self, inputs, targets):
        loss_ce = self.ce_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        combined_loss = self.weight_ce * loss_ce + self.weight_dice * loss_dice
        return combined_loss
