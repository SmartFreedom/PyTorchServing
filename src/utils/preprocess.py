import numpy as np
import cv2
import scipy.ndimage

from src.configs import config
from src.api import queue_manager as qm


class Process:
    @staticmethod
    def process(data, manager: qm.QueueManager=None):
        return data

class MammographyRoIProcess(Process):
    @staticmethod
    def process(data, manager: qm.QueueManager=None):
        processed = data.copy()

        v = data['image']
        if config.PROCESS.RoI.CROP_SIDE is not None:
            shape = np.array(v.shape)[:2]
            shape_ = (shape
                      * (config.PROCESS.RoI.CROP_SIDE
                         / shape.min())).astype(np.int)
            v = cv2.resize(v, tuple(shape_[::-1].tolist()))
            processed['image'] = v
        return processed


def resize_image(image, interpolation=2, side=None):
    if side is None:
        return image
    if image is None:
        return None
    shape = np.array(image.shape)[:2]
    shape_ = (shape * (side / shape.max())).astype(np.int)
    return cv2.resize(
        image, tuple(shape_[::-1].tolist()), interpolation=interpolation)


class MammographyMassProcess(Process):
    @staticmethod
    def process(data, manager: qm.QueueManager):
        processed = data.copy()

        key = config.API.PID_SIDE2KEY(data['channel'], data['side'])
        shape = np.array(data['image'].shape[:2])
        roi = manager.MammographyRoI.predictions[key]['whole']

        
        roi, colors = scipy.ndimage.label(roi > .02)
        roi = roi == np.argmax(np.bincount(roi[roi != 0]))
        processed['image'] = resize_image(
            data['image'], side=config.PROCESS.MASS.MAX_SIDE)
        roi = cv2.resize(
            roi.astype(np.uint), 
            processed['image'].shape[::-1], 
            interpolation=0)
        processed['mask'] = scipy.ndimage.binary_dilation(roi, iterations=100)
        processed['scale_factor'] = shape / np.array(roi.shape[:2])
        return processed
