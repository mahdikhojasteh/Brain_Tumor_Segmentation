import torch
import torch.nn as nn
import torch.nn.functional as F


ALPHA = 0.8
GAMMA = 2

class FocalLossMulticlass(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
                
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        # inputs    (b, 4, 192, 192) -> (b, 4, 192, 192)
        inputs = F.softmax(inputs, dim=1)
        
        # targets   (b, 4, 192, 192) -> (b, 192, 192)
        targets = targets.argmax(dim=1)
        
        #first compute binary cross-entropy 
        BCE = F.cross_entropy(inputs, targets)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss