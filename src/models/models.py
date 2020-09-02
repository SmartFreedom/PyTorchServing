from src.modules import lungs_segmentation as lsg
from src.modules import learner as lrn
from src.modules import dataset as ds
from src.utils import ct_reader as ctr
from src.configs import config

import numpy as np


class Models:
    def __init__(self):
        self.collection = dict()

    def get_model(self, item: str, version=None):
        assert item in config.MODELS

        checkpoint = config.PATHS.MODELS / item / config.MODELS[item]['path']
        kwargs = config.MODELS[item]['kwargs']
        if 'pytorch' not in config.MODELS[item]['type']:
            kwargs['checkpoint'] = checkpoint
            return config.MODELS[item]['model'](**kwargs)

        model = config.MODELS[item]['model'](**kwargs)
        model = lrn.to_single_channel(model, config.MODELS[item]['fc'])
        model = lrn.get_model(model, checkpoint=checkpoint, devices=config.DEVICES)

        if 'inference' in config.MODELS[item].keys():
            return config.MODELS[item]['inference'](model)
        else: return lrn.Inference(model) 

    def __getitem__(self, item):
        if item in self.collection.keys():
            config.API.LOG('{} model already exists!'.format(item))
            return self.collection[item]
        self.collection[item] = self.get_model(item)
        config.API.LOG('{} model has been created!'.format(item))
        return self.collection[item]
