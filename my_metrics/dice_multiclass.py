import torch.nn as nn
import torch.nn.functional as F

# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook

class DiceLossMulticlass(nn.Module):
    def __init__(self, weights=None, calculate_weight=False):
        super().__init__()
        self.weights = weights
        self.calculate_weight = calculate_weight

    def forward(self, inputs, targets, smooth=1):
        if self.weights is not None:
            assert self.weights.shape == (targets.shape[1], )

        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction images, leave BATCH and NUM_CLASSES
        # (BATCH, NUM_CLASSES, H, W) -> (BATCH, NUM_CLASSES, H * W)
        inputs = inputs.view(inputs.shape[0],inputs.shape[1],-1)
        targets = targets.view(targets.shape[0],targets.shape[1],-1)

        #intersection = (inputs * targets).sum()
        intersection = (inputs * targets).sum(0).sum(1)
        #dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        dice = (2.*intersection + smooth)/(inputs.sum(0).sum(1) + targets.sum(0).sum(1) + smooth)

        if (self.weights is None) and self.calculate_weight==True:
            self.weights = (targets == 1).sum(0).sum(1).float()
            self.weights /= self.weights.sum() # so they sum up to 1

        if self.weights is not None:
            return 1 - (dice*self.weights).mean()
        else:
            return 1 - dice.mean()