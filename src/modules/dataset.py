import os
import cv2
import numpy as np
import easydict

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.transforms import ToTensor, Normalize, Compose

from src.configs import config
import src.modules.augmentations as augs


def inference_collater(data):
    pids = [ pid for s in data for pid in [s['pid']] * 8 ]
    sides = [ side for s in data for side in [s['side']] * 8 ]
    shapes = [ pid for s in data for pid in [s['shape']] * 8 ]
    imgs = [ s['image'] for s in data ]
    _shapes = np.array([ s.shape[1:] for s in imgs ])
    max_shape = _shapes.max() 
    max_shape += (128 - max_shape % 128) * (max_shape % 128 != 0)

    imgs = [np.pad(
        s, 
        ((0, 0),
        (0, max_shape - s.shape[1]),
        (0, max_shape - s.shape[2])),
        mode='constant',
        constant_values=np.median(s)
    ) for s in imgs ]
    imgs = torch.stack([ 
        torch.Tensor(im.copy())
        for s in imgs for im in augs._rotate_mirror_do(s) ])
    
    return { 'image': imgs, 'pid': pids, 'shape': shapes, 'side': sides }


def infer_on_batch(model, data):
    batch = [data['image']]
    if np.array(data['shape']).max() > 1000:
        batch = [ torch.unsqueeze(d, 0) for d in data['image'] ]

    torch.cuda.empty_cache()
    predict = list()
    for image in batch:
        image = torch.autograd.Variable(image).cuda()
        predict.append(model(image).sigmoid().data.cpu().numpy())
        image = image.data.cpu().numpy()
        torch.cuda.empty_cache()
    predict = np.squeeze(np.stack(predict))
    return predict if len(predict.shape) == 4 else np.squeeze(predict, axis=0)


class InferenceDataset(Dataset):
    def __init__(self, queue, transform):
        """queue: qm.LQueue"""
        self.img_transform = transform
        self.queue = queue

    def __getitem__(self, idx):
        return self.process(self.queue[idx])

    def process(self, data):
        data = {
            "image": data['image'],
            "pid": data['channel'],
            "side": data['side'],
            "mask": data['mask'][np.newaxis] if 'mask' in data.keys() else np.nan,
            "scale_factor": data['scale_factor'] if 'scale_factor' in data.keys() else np.nan,
            "shape": data['image'].shape
        }
        data = easydict.EasyDict(data)
        data.image = self.img_transform(np.expand_dims(data.image, -1))
        return data

    def __len__(self):
        return len(self.queue)


class InferenceIterableDataset(IterableDataset):
    def __init__(self, queue):
        """queue: qm.LQueue"""
        self.queue = queue
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=config.PROCESS.RoI.MEAN, std=config.PROCESS.RoI.STD)
        ])

    def __iter__(self):
        return DatasetIterator(self)

    def process(self):
        data = self.queue.pop()
        if data is None:
            raise StopIteration

        data = {
            "image": data['image'],
            "pid": data['channel'],
            "side": data['side'],
            "shape": data['image'].shape
        }
        data = easydict.EasyDict(data)
        data.image = self.img_transform(np.expand_dims(data.image, -1))
        return data


class DatasetIterator:
    def __init__(self, dataset: InferenceIterableDataset):
        self.dataset = dataset

    def __next__(self):
        return self.dataset.process()


class TDataset(Dataset):
    def __init__(self, augmentations=None):
        self.transform = augmentations
        self.scan = list()
        self.names = list()

    def __getitem__(self, idx):
        image = self.scan[idx]

        data = {"image": image}
        if self.transform is not None:
            data = self.transform(data)
        data.update({
            "idx": idx,
            "fileid": self.names[idx]
        })
        return self.postprocess(**data)

    def postprocess(self, **kwargs):
        pass

    def preprocess(self, **kwargs):
        pass

    def populate(self, names, **kwargs):
        # **kwargs == lungs_mask=None
        self.scan = self.preprocess(**kwargs)
        self.names = names

    def __len__(self):
        return len(self.scan)


class CTDataset(TDataset):
    def __init__(self, augmentations=None):
        super().__init__(augmentations)
        self.to_tensor = Compose([
            ToTensor(),
            Normalize(
                mean=config.CT_PARAMS.MEAN,
                std=config.CT_PARAMS.STD
            )
        ])

    def postprocess(self, **kwargs):
        kwargs.update({
            'image': self.to_tensor(kwargs['image'].astype(np.uint8)),
        })
        return kwargs

    def preprocess(self, scan, lungs=None):
        scan += 1024
        scan = 255. * np.clip(scan, 0, 500) / 500.  # -> [0-255]
        if lungs is not None:
            scan[lungs == 0] = config.CT_PARAMS.MEAN
        return scan


class MammographyDataset(TDataset):
    def __init__(self, augmentations=None):
        super().__init__(augmentations)
        self.to_tensor = Compose([
            ToTensor(),
            Normalize(
                mean=config.MAMMOGRAPHY_PARAMS.MEAN,
                std=config.MAMMOGRAPHY_PARAMS.STD
            )
        ])

    def postprocess(self, **kwargs):
        kwargs = easydict.EasyDict(kwargs)
        kwargs.image = self.to_tensor(np.expand_dims(kwargs['image'], -1))
        return kwargs

    def load_image(self, fileid):
        return cv2.imread(os.path.join(self.root, fileid), 0)
