import os
import cv2
import numpy as np
import easydict

from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.transforms import ToTensor, Normalize, Compose

from src.api.queue_manager import QueueManager
from src.configs import config


img_transform = Compose([
    ToTensor(),
    Normalize(mean=config.MEAN, std=config.STD)
])


class InferenceDataset(IterableDataset):
    def __init__(self, queue: QueueManager.LQueue):
        self.queue = queue

    def __iter__(self):
        return DatasetIterator(self)

    def process(self):
        data = self.queue.pop()
        data = {
            "image": data['image'],
            "pid": data['channel'],
            "shape": data['image'].shape
        }
        data = easydict.EasyDict(data)
        data.image = img_transform(np.expand_dims(data.image, -1))
        return data


class DatasetIterator:
    def __init__(self, dataset: InferenceDataset):
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
