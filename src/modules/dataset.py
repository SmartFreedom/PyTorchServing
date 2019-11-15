import os
import cv2
import numpy as np
import pandas as pd
import sklearn.model_selection
import skimage
from glob import glob

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

from ..configs import config
from ..utils import ct_reader as ctr


img_transform = Compose([
    ToTensor(),
    Normalize(mean=config.MEAN, std=config.STD)
])


class CTDataset(Dataset):
    def __init__(self, augmentations=None):
        self.transform = augmentations

    def __getitem__(self, idx):
        image = self.scan[idx]

        data = { "image": image }
        if self.transform is not None:
            data = self.transform(data)
        data.update({
            "idx": idx ,
            "fileid": self.names[idx]
        })
        return self.postprocess(**data)

    def postprocess(self, image, **kwargs):
        kwargs.update({
            'image': img_transform(image.astype(np.uint8)),
        })
        return kwargs

    def preprocess(self, image, lungs=None):
        image += 1024
        image = 255. * np.clip(image, 0, 500) / 500. # -> [0-255]
        if lungs is not None:
            image[lungs == 0] = config.MEAN
        return image

    def populate(self, names, scan, lungs_mask=None):
        self.scan = self.preprocess(scan, lungs_mask)
        self.names = names

    def __len__(self):
        return len(self.scan)

