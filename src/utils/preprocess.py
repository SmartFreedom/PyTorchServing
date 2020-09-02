import sklearn
import sklearn.mixture

import numpy as np
import cv2
import scipy.ndimage

from src.configs import config
from src.api import queue_manager as qm


def get_mixture_stats(image):
    median = np.median(image)
    c1 = (image == median).sum() / (image == median + 1).sum() ** .5
    c2 = (image == median).sum() / (image == median - 1).sum() ** .5

    # if min(c1, c2) > 10:
    try:
        GM = sklearn.mixture.GaussianMixture(
            n_components=2, 
            covariance_type='spherical')
        GM.fit(image.flatten().reshape((-1, 1)))
        image_mean = GM.means_[np.argmax(GM.covariances_)]
        back_mean = GM.means_[np.argmin(GM.covariances_)]
        image_std = np.sqrt(GM.covariances_).max()
    except:
        GM = sklearn.mixture.GaussianMixture(
        n_components=1, 
        covariance_type='spherical')
        GM.fit(image[image != median].reshape((-1, 1)))
        image_mean = GM.means_[np.argmax(GM.covariances_)]
        image_std = np.sqrt(GM.covariances_).max()
        back_mean = median
    return { 
        'image_mean': image_mean, 
        'image_std': image_std, 
        'back_mean': back_mean 
    }


def get_mask_stats(image, mask):
    mask = cv2.resize(
        mask.astype(np.uint8), 
        image.shape[::-1], 
        interpolation=0) > 0
    back_mean = np.median(image[np.logical_not(mask)])
    roi = image[mask]
    image_mean = np.median(roi)
    percentiles = [5., 0.]
    if back_mean > image_mean: percentiles = percentiles[::-1]
    image_std = np.std(roi[
        (roi > np.percentile(roi, percentiles[0])) 
        & (roi < np.percentile(roi, 100 - percentiles[1]))
    ])
    back_mean > image_mean
    return { 
        'image_mean': image_mean, 
        'image_std': image_std, 
        'back_mean': back_mean 
    }


def convert_and_unify(image, mask=None):
    if mask is not None:
        image_mean, image_std, back_mean = list(get_mask_stats(image, mask).values())
    else:
        image_mean, image_std, back_mean = list(get_mixture_stats(image).values())

    mid = 2 ** 8 if image.dtype == np.uint8 else 2 ** 16
    if back_mean > image_mean:
        image = mid - 1 - image
        image_mean = mid - 1 - image_mean

    if image.dtype != np.uint8:
        image = (image.astype(np.float) - image_mean) / image_std 
        image = image \
        * np.array(config.MAMMOGRAPHY_PARAMS.INNER_STD) \
        + np.array(config.MAMMOGRAPHY_PARAMS.INNER_MEAN)
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    return image


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
            if v.dtype == np.uint16:
                v = convert_and_unify(v)
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
