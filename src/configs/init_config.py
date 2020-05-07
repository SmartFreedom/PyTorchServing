from src.models.models import Models, Preprocess
from src.configs import config
import utils

import torch
import os


utils.make_dirs()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
torch.cuda.set_device(config.CUDA_IDX)

models = Models()
preprocess = Preprocess()
