from pathlib import Path
import os
import easydict
import addict


DEVICES = [0, 1]

PATHS = easydict.EasyDict()
PATHS.DATA_ROOT = Path(os.getcwd()) / '..' / 'data'
PATHS.MODELS = PATHS.DATA_ROOT / 'models'
PATHS.RESULTS = PATHS.DATA_ROOT / 'results'
PATHS.LOGDIR = PATHS.DATA_ROOT / 'logdir'
PATHS.OUTPUT = PATHS.DATA_ROOT / 'output'

BATCH_SIZE = 1
WORKERS_NB = 8

CUDA_VISIBLE_DEVICES = "1, 2"
CUDA_IDX = 0


from src.configs import ct_config, mammography_config, init_config, api_config, process_config


CT_PARAMS = easydict.EasyDict(ct_config.PARAMS.copy())
MAMMOGRAPHY_PARAMS = mammography_config.PARAMS.copy()
PROCESS = addict.Dict(process_config.PROCESS.copy())
SHARED = easydict.EasyDict(init_config.PARAMS.copy())
API = easydict.EasyDict(api_config.API.copy())

# set the URL where you can download your model weights
MODELS = dict()
MODELS.update(ct_config.MODELS)
MODELS.update(mammography_config.MODELS)

THRESHOLDS = {
    'fpn': [.5] * 2,
    'head': [.5] * 5,
}

THRESHOLDS_LOWER_BOUND = {
    'fpn': [.3] * 2,
    'head': [.3] * 5,
}
