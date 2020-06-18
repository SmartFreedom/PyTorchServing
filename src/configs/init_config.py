from src.models.models import Models, Preprocess
from src.configs import config
from src.utils import utils
from src.utils import preprocess as ps

import torch
import os
import easydict


def init():
    utils.make_dirs()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    torch.cuda.set_device(config.CUDA_IDX)

    config.SHARED.models = Models()
    config.SHARED.preprocess = Preprocess()
    config.PROCESS.MAP = {
        ps.MammographyRoIProcess: [
            'MammographyRoI', 
            'DensityEstimation', 
            'AsymmetryEstimation'
        ],
        ps.MammographyMassProcess: [
            'MassSegmentation',
        ],
    }


PARAMS = easydict.EasyDict()
PARAMS.INIT = init
