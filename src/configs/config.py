from pathlib import Path
import os
import easydict

from src.configs import ct_config, mammography_config, init_config, api_config, process_config


CT_PARAMS = easydict.EasyDict(ct_config.PARAMS.copy())
MAMMOGRAPHY_PARAMS = easydict.EasyDict(mammography_config.PARAMS.copy())
PROCESS = easydict.EasyDict(process_config.PROCESS.copy())
SHARED = easydict.EasyDict(init_config.PARAMS.copy())
API = easydict.EasyDict(api_config.API.copy())

DEVICES = [0]

PATHS = easydict.EasyDict()
PATHS.DATA_ROOT = Path(os.getcwd()) / 'data'
PATHS.MODELS = PATHS.DATA_ROOT / 'models'
PATHS.RESULTS = PATHS.DATA_ROOT / 'results'
PATHS.LOGDIR = PATHS.DATA_ROOT / 'logdir'

# set the URL where you can download your model weights
MODELS = dict()
MODELS.update(ct_config.MODELS)
MODELS.update(mammography_config.MODELS)

BATCH_SIZE = 8

CUDA_VISIBLE_DEVICES = "0"
CUDA_IDX = 0
