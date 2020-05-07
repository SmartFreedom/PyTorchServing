from src.configs import config
from pathlib import Path
import os
import requests


# set dir structure
def make_dirs():
    os.makedirs(config.PATHS.DATA_ROOT, exist_ok=True)
    os.makedirs(config.PATHS.RESULTS, exist_ok=True)
    os.makedirs(config.PATHS.MODELS, exist_ok=True)

    for model in config.MODELS.keys():
        os.makedirs(os.path.join(config.PATHS.MODELS, model), exist_ok=True)


def get_model_url():
    # download model weights if not already saved
    path_to_model = os.path.join(config.DATA_ROOT, 'models', config.MODEL_NAME)
    if not os.path.exists(path_to_model):
        print('done!\nmodel weights were not found, downloading them...')

        filename = Path(path_to_model)
        r = requests.get(config.MODEL_URL)
        filename.write_bytes(r.content)

    print('done!\nloading up the saved model weights...')
