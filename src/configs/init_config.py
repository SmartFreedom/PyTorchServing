from src.models.models import Models, Preprocess
from src.configs import config
from src.utils import utils

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

PARAMS = easydict.EasyDict()
PARAMS.INIT = init
