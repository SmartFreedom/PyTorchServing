import numpy as np
import cv2

from src.configs import config


class Process:
    @staticmethod
    def process(data):
        return data

class MammographyRoIProcess(Process):
    @staticmethod
    def process(data):
        processed = dict()
        for k, v in data.items():
            if config.PROCESS.RoI.CROP_SIDE is not None:
                shape = np.array(v.shape)[:2]
                shape_ = (shape
                          * (config.PROCESS.RoI.CROP_SIDE
                             / shape.min())).astype(np.int)
                v = cv2.resize(v, tuple(shape_[::-1].tolist()))
            processed[k] = v
        return processed
