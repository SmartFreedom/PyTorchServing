import torch
from torch import nn

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import src.modules.augmentations as augs
import src.modules.smooth_tile_predictions as smt
import src.utils.maxima_finder as mf
from src.configs import config


def get_model(model, checkpoint=None, map_location=torch.device(config.CUDA_IDX), devices=None):
    model.cuda()

    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location)  # .module.state_dict()
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd}
        print('Overlapped keys: {}'.format(len(sd.keys())))
        msd.update(sd)
        model.load_state_dict(msd)

    if devices is not None:
        model = torch.nn.DataParallel(model, device_ids=devices)

    return model


def to_single_channel(model, fc=None):
    if fc is not None:
        model.fc = fc
    try:
        return albunet_to_single_channel(model)
    except:
        pass
    try:
        return resnet_to_single_channel(model)
    except:
        raise('Error')


def resnet_to_single_channel(model):
    conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    nstd = {
        key: param.sum(1).unsqueeze(1)
        for key, param in model.conv1.state_dict().items()
    }
    print('Summed over: {}'.format(' | '.join(nstd.keys())))

    conv1.load_state_dict(nstd)
    model.conv1 = conv1
    return model


def albunet_to_single_channel(model):
    conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    nstd = {
        key: param.sum(1).unsqueeze(1)
        for key, param in model.conv1[0].state_dict().items()
    }
    print('Summed over: {}'.format(' | '.join(nstd.keys())))

    conv1.load_state_dict(nstd)
    model.conv1[0] = conv1
    model.encoder.conv1 = model.conv1[0]
    return model


def freeze(model, unfreeze=False):
    children = list(model.children())
    if hasattr(model, 'children') and len(children):
        for child in children:
            freeze(child, unfreeze)
    elif hasattr(model, 'parameters'):
        for param in model.parameters():
            param.requires_grad = unfreeze


def unfreeze_bn(model):
    predicat = isinstance(model, torch.nn.BatchNorm2d)
    predicat |= isinstance(model, bn.ABN)
    predicat |= isinstance(model, bn.InPlaceABN)
    predicat |= isinstance(model, bn.InPlaceABNSync)
    if predicat:
        for param in model.parameters():
            param.requires_grad = True

    children = list(model.children())
    if len(children):
        for child in children:
            unfreeze_bn(child)
    return None


class Inference:
    def __init__(self, model):
        self.model = model

    def make_step(self, data, training=False):
        image = self.format_input(data)
        prediction = self.model(image)
        results = self.format_output(prediction, data)
        image = image.data.cpu()

        return results

    def validate(self, datagen):
        torch.cuda.empty_cache()
        self.model.eval()
        meters = defaultdict(list)
        results = dict()

        with torch.no_grad():
            for data in tqdm(datagen):
                results.update(
                    self.make_step(data, training=False))
                # meters = self._format_meters(meters, data)

        return results

    @staticmethod
    def _format_input(data):
        image = torch.autograd.Variable(data['image']).cuda().float()

        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)

        return image

    def format_input(self, data):
        return self._format_input(data)

    @staticmethod
    def _format_output(prediction, data):
        results = {
            k: v.data.cpu() if torch.is_tensor(v) else v 
            for k, v in data.items() if k != 'image'
        }
        results.update({
            'prediction': prediction.data.cpu() 
            if torch.is_tensor(prediction) else prediction})
        return results

    def format_output(self, prediction, data):
        return self._format_output(prediction, data)

    @staticmethod
    def _format_meters(meters, data):
        for k, v in data.items():
            if torch.is_tensor(v):
                v = v.data.cpu().numpy()
            meters[k].append(v)
        return meters

    @staticmethod
    def _meters_to_array(meters):
        for k, v in meters.items():
            meters[k] = np.array(v)
        return meters


class InferenceDensity(Inference):
    def format_output(self, prediction, data):
        data = self._format_output(prediction, data)
        return {
            config.API.PID_SIDE2KEY(data['pid'][0], data['side'][0]):
            {
                'density': data['prediction'],
            }
        }

    @staticmethod
    def _format_output(prediction, data):
        results = {
            k: v.data.cpu() if torch.is_tensor(v) else v 
            for k, v in data.items() if k != 'image'
        }
        results.update({'prediction': prediction.softmax(1).data.cpu().numpy()})
        return results


class InferenceRoI(Inference):
    def format_output(self, prediction, data):
        data = self._format_output(prediction, data)
        prediction = augs._rotate_mirror_undo(data['prediction'])
        shape = data['shape'][0]
        p = prediction[..., :shape[0], :shape[1]]
        return {
            config.API.PID_SIDE2KEY(data['pid'][0], data['side'][0]):
            {
                'tissue': p[0],
                'whole': p[1],
                'nipple': p[2],
            }
        }

    @staticmethod
    def _format_output(prediction, data):
        results = {
            k: v.data.cpu() if torch.is_tensor(v) else v 
            for k, v in data.items() if k != 'image'
        }
        results.update({'prediction': prediction.sigmoid().data.cpu().numpy()})
        return results


class InferenceMass(Inference):
    def format_input(self, data):
        return self._format_input(data)

    @staticmethod
    def format_output(data):
        results = {
            k: v.data.cpu() if torch.is_tensor(v) else v 
            for k, v in data.items() if k != 'image'
        }
        results = {}
        data['centroids'] = list()
        roi = data['mask'].data.numpy().astype(np.uint)[:, 0]
        for i, pid in enumerate(data['pid']):
            roi_ = cv2.resize(
                roi[i], data['fpn_predictions'].shape[2:][::-1], interpolation=0)
            data['fpn_predictions'][i] = roi_[np.newaxis] * data['fpn_predictions'][i]
            roi_ = cv2.resize(
                roi[i], data['head_predictions'].shape[2:][::-1], interpolation=0)
            data['head_predictions'][i] = roi_[np.newaxis] * data['head_predictions'][i]

            seriesuid = config.API.PID_SIDE2KEY(pid, data['side'][i])
            results.update({
                config.API.PID_SIDE2KEY(data['pid'][i], data['side'][i]): {
                    'fpn_predictions': data['fpn_predictions'][i],
                    'head_predictions': data['head_predictions'][i],
                    'centroids': pd.DataFrame(mf.find_maxima(
                        data['fpn_predictions'][i][0],
                        seriesuid=seriesuid, 
                        scale_factor=data['scale_factor'][i] * 2))
            }})

        return results

    def make_step(self, data, training=False, verbose=False):
        results = list()
        data['head_predictions'] = list()
        data['fpn_predictions'] = list()
        for img in np.rollaxis(data['image'].data.numpy(), 1, len(data['image'].shape)):
            fpn_predictions, head_predictions = self.perform(img, verbose=False)
            data['fpn_predictions'].append(fpn_predictions)
            data['head_predictions'].append(head_predictions)
        data['fpn_predictions'] = np.array(data['fpn_predictions'])
        data['head_predictions'] = np.array(data['head_predictions'])
        return self.format_output(data)

    def perform(self, input_img, verbose=False):
        pad = smt._pad_img(input_img, config.PROCESS.MASS.WINDOW_SIDE, 
                           config.PROCESS.MASS.SUB_DIVISIONS)
        pads = smt._rotate_mirror_do(pad)

        head_predictions = list()
        fpn_predictions = list()
        for pad in pads:
            # For every rotation:
            fpn_prediction, head_prediction = smt._windowed_subdivs(
                pad, config.PROCESS.MASS.WINDOW_SIDE, 
                config.PROCESS.MASS.SUB_DIVISIONS, self.model)

            shape = np.array(pad.shape[:-1])
            head_predictions.append(smt._recreate_from_subdivs(
                head_prediction, config.PROCESS.MASS.WINDOW_SIDE//32, 
                config.PROCESS.MASS.SUB_DIVISIONS,
                padded_out_shape=(head_prediction.shape[2], *(shape // 32).tolist())))

            fpn_predictions.append(smt._recreate_from_subdivs(
                fpn_prediction, config.PROCESS.MASS.WINDOW_SIDE//2, 
                config.PROCESS.MASS.SUB_DIVISIONS,
                padded_out_shape=(fpn_prediction.shape[2], *(shape//2).tolist())))

        # Merge after rotations:
        fpn_predictions = smt._rotate_mirror_undo(fpn_predictions)
        head_predictions = smt._rotate_mirror_undo(head_predictions)

        fpn_predictions = smt._unpad_img(
            fpn_predictions, config.PROCESS.MASS.WINDOW_SIDE//2, 
            config.PROCESS.MASS.SUB_DIVISIONS)
        head_predictions = smt._unpad_img(
            head_predictions, config.PROCESS.MASS.WINDOW_SIDE//32, 
            config.PROCESS.MASS.SUB_DIVISIONS)

        if verbose:
            clear_output(wait=True)
            smt.plot_infer(image, fpn_predictions, head_predictions)

        return fpn_predictions, head_predictions
