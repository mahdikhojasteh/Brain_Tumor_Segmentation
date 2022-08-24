import torch.nn as nn
import torch.nn.functional as F

class IoULossMulticlass(nn.Module):
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
        #total = (inputs + targets).sum()
        total = (inputs + targets).sum(0).sum(1)
        union = total - intersection 
        #IoU = (intersection + smooth)/(union + smooth)
        IoU = (intersection + smooth)/(union + smooth)

        if (self.weights is None) and self.calculate_weight==True:
            self.weights = (targets == 1).sum(0).sum(1)
            self.weights /= self.weights.sum() # so they sum up to 1

        if self.weights is not None:
            return 1 - (IoU*self.weights).mean()
        else:
            return 1 - IoU.mean()