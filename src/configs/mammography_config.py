import easydict
import addict
import numpy as np
from torch import nn
from torchvision.transforms import ToTensor, Normalize, Compose

import src.modules.learner as lrn
from src.models.albunet import AlbuNet, getFPNAlbuNetSE
from src.models.mass_segm_albunet import AlbuNet as AlbuNetHEAD
from src.models.resnet import resnet34
from src.models.regression_tree import ProbabilityClassifier


PARAMS = addict.Dict()
PARAMS.MEAN = [74.77618355029848]
PARAMS.STD = [31.738553261533994]

PARAMS.INNER_MEAN = [0.24830265228665568]
PARAMS.INNER_STD = [0.13130028764906437]

PARAMS.BIRADS_CENTROIDS = [0.1, 0.35, 0.65, 0.85]

PARAMS.DEFAULT_THRESHOLDS = {
    'radiant_node': .5,
    'intramammary_lymph_node': .12,
    'calcification': .5,
    'mask': .5,
    'structure': .5,
    'border': .5,
    'shape': .5,
    'malignancy': .5,
    'calcification_malignant': .5,
    'local_structure_perturbation': .5,
}

MODELS = {
    'MammographyRoI': {
        'model': AlbuNet,
        'type': 'pytorch',
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
        'path': 'albunet18_with_augs_fold_2_epoch_249_best.pth',
    },
    'DensityEstimation': {
        'model': resnet34,
        'type': 'pytorch',
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
        'type': 'pytorch',
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
        'model': AlbuNetHEAD,
        'type': 'pytorch',
        'transform': lambda x: np.rollaxis(x, -1, 0),
        'inference': lrn.InferenceMass,
        'fc': None,
        'kwargs': {
            'num_classes': 2, 
            'num_head_classes': 7, 
            'is_deconv': True, 
            'dropout': .25,
        },
        'path': 'albunet34_fold_0_250thth_best.pth',
    },
    'DecisionTreeClassifier': {
        'model': ProbabilityClassifier,
        'type': 'sklearn',
        'transform': lambda x: np.rollaxis(x, -1, 0),
        'inference': lrn.InferenceMass,
        'fc': None,
        'kwargs': {},
        'path': 'XGBOOST.pth',
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
        5: 'radiant_node',
        6: 'intramammary_lymph_node',
    }
}

PARAMS.REVERSE_NAMES = {
    'fpn': {
        'calcification': 0,
        'mask': 1,
    },
    'head': {
        'structure': 0,
        'border': 1,
        'shape': 2,
        'calcification_malignant': 3,
        'local_structure_perturbation': 4,
        'radiant_node': 5,
        'intramammary_lymph_node': 6,
    }
}

PARAMS.MODES = { 
    'mass': [
        'structure', 'border', 'shape', 
        'local_structure_perturbation', 'radiant_node'
    ],
    'lymph_node': ['intramammary_lymph_node'],
    'calcification': []
}

PARAMS.POSITIVE_MAPS = {
    'structure': 'inhomogen', 
    'border': 'obscure_margin',
    'shape': 'irregular_shape',
    'radiant_node': 'spiculate_margin',
    'intramammary_lymph_node': 'lymph_node',
}

PARAMS.NEGATIVE_MAPS = {
    'structure': 'homogen', 
    'border': 'circ_margin',
    'shape': 'regular_shape',
    'radiant_node': 'spiculate_margin',
    'intramammary_lymph_node': 'lymph_node',
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
