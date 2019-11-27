from src.modules import lungs_segmentation as lsg
from src.modules import learner as lrn
from src.modules import dataset as ds
from src.models.unet import TernausNet
from src.utils import ct_reader as ctr
from src.configs import config

import numpy as np


class Models:
    def __init__(self):
        self.model_init = {
            'AlbuNet': {
                'model': TernausNet.AlbuNet,
                'kwargs': {
                    'num_classes': 1,
                    'is_deconv': False
                },
                'RetinaNet': {}
            },
        }
        self.collection = dict()

    def get_model(self, item, version=None):
        assert item in config.MODELS

        model = self.model_init[item]['model'](**self.model_init[item]['kwargs'])
        model = lrn.to_single_channel(model)
        checkpoint = config.PATHS.MODELS / item / config.MODELS[item]
        model = lrn.get_model(model, checkpoint=checkpoint, devices=config.DEVICES)
        return lrn.Inference(model)

    def __getitem__(self, item):
        if item in self.collection.keys():
            print('{} model already exists!'.format(item))
            return self.collection[item]
        self.collection[item] = self.get_model(item)
        print('{} model has been created!'.format(item))
        return self.collection[item]


# TODO: refactor as Models class
class Preprocess:
    def __init__(self):
        self.dataset = ds.CTDataset()

    def __call__(self, item, inpt):
        assert item in config.MODELS
        names, dscan = ctr.read_ct_scan(inpt)
        scan = ctr.get_pixels_hu(dscan)

        # lungs, lung_left, lung_right, trachea = lsg.improved_lung_segmentation(scan)
        # segmentation = np.max([3 * lung_left, 2 * lung_right, 1 * trachea], axis=0)
        self.dataset.populate(names, scan, None)#segmentation)
        return ds.DataLoader(self.dataset, batch_size=config.BATCH_SIZE)
        # lsg.save_lungs_mask(segmentation, names, case)
