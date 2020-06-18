import easydict
import addict
import numpy as np
from torch import nn
from torchvision.transforms import ToTensor, Normalize, Compose

import src.modules.learner as lrn
from src.models.albunet import AlbuNet
from src.models.albunet import getFPNAlbuNetSE
from src.models.resnet import resnet34

PARAMS = addict.Dict()
PARAMS.MEAN = [74.77618355029848]
PARAMS.STD = [31.738553261533994]

MODELS = {
    'MammographyRoI': {
        'model': AlbuNet,
        'transform': Compose([
            ToTensor(),
            Normalize(mean=PARAMS.MEAN, std=PARAMS.STD)
        ]),
        'fc': None,
        'inference': lrn.InferenceRoI,
        'kwargs': {
            'num_classes': 3,
            'is_deconv': True,
            'dropout': .2,
        },
        'path': 'albunet18_fold_2_best.pth',
    },
    'DensityEstimation': {
        'model': resnet34,
        'transform': Compose([
            ToTensor(),
            Normalize(mean=PARAMS.MEAN, std=PARAMS.STD)
        ]),
        'inference': lrn.InferenceDensity,
        'fc': nn.Linear(in_features=512, out_features=4, bias=True),
        'kwargs': {
            'dropout': .8, 
            'dropout2d': .8
        },
        'path': 'resnet34_fold_0_best.pth',
    },
    'AsymmetryEstimation': {
        'model': resnet34,
        'transform': Compose([
            ToTensor(),
            Normalize(mean=PARAMS.MEAN, std=PARAMS.STD)
        ]),
        'inference': lrn.Inference,
        'fc': nn.Linear(in_features=512, out_features=4, bias=True),
        'kwargs': {
            'dropout': .8, 
            'dropout2d': .8
        },
        'path': 'resnet34_fold_0_best.pth',
    },
    'MassSegmentation': {
        'model': getFPNAlbuNetSE,
        'transform': lambda x: np.rollaxis(x, -1, 0),
        'inference': lrn.InferenceMass,
        'fc': None,
        'kwargs': {
            'num_classes': 2, 
            'num_head_classes': 5, 
            'is_deconv': False, 
            'dropout': .25,
        },
        'path': 'head_fpn_albunet_se50x_fold_0_100th_epoch51_best.pth',
    }
}

PARAMS.NAMES = {
    'fpn': {
        0: 'calcification',
        1: 'mask',
    },
    'head': {
        0: 'structure',
        1: 'border',
        2: 'shape',
        3: 'calcification_malignant',
        4: 'local_structure_perturbation',
    }
}

PARAMS.MODELS = MODELS
PARAMS.CROP_SIDE = 1024
PARAMS.CNN_SIDE = 512
PARAMS.CROP_STEP = 256
PARAMS.SIDE = 512
PARAMS.DROPOUT = .2

PARAMS.MIN_DIAMETER = 15
PARAMS.MAX_DIAMETER = 108

PARAMS.TARGET_NAME = 'label'

PARAMS.ASSYMETRY_TYPES = [
    "Нет",
    "Есть Локальная (очаговая)",
    "Есть Тотальная",
    "Есть Локальная с кальцинатами",
    "Есть Динамическая"
]

PARAMS.DENSITY_TYPES = [
    'A',
    'B',
    'C',
    'D'
]
