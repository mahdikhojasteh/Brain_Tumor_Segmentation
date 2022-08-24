import os
import numpy as np
import torch
from torch.utils.data import Dataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class BrainDataset(Dataset):
    def __init__(self, img_pathes, msk_pathes, transform=None):
        self.img_pathes = img_pathes
        self.msk_pathes = msk_pathes
        self.transform = transform

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, index):
        image = np.load(self.img_pathes[index], allow_pickle=True)
        mask = np.load(self.msk_pathes[index], allow_pickle=True)

        image = image.astype(np.float32)
        mask = mask.astype(np.uint8)

        if self.transform is not None:
            image = np.transpose(image, (1, 2, 0))
            mask = np.transpose(mask, (1, 2, 0))
            mask = SegmentationMapsOnImage(mask, mask.shape)
            image, mask_aug = self.transform(image=image, segmentation_maps=mask)
            mask = mask_aug.get_arr()
            image = np.transpose(image, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask
    

