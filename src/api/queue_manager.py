from collections import defaultdict
import multiprocessing as mp
import queue as eq
import easydict
import imageio
import requests
import pydicom as dicom
import torch
import numpy as np
import gc
import os

from src.configs import config
import src.api.redis as rd
import src.api.response as rs
import src.modules.dataset  as ds
import src.utils.preprocess as ps


def load_image(url):
    r = requests.get(url, allow_redirects=True)
    dfile = dicom.filebase.DicomBytesIO(r.content)
    return dicom.read_file(dfile).pixel_array


class QueueManager(easydict.EasyDict):
    def __init__(self, r_api: rd.RedisAPI, mp_queue: mp.Queue):
        super(QueueManager, self).__init__()
        self.mp_queue = mp_queue
        self.r_api = r_api
        self.queue = list()
        self.keys = config.API.KEYS
        self.update({k: LQueue(self, k) for k in self.keys})

    def start(self):
        self.clear()
        config.API.LOG('Check status...')
        self.check_status()
        self.process([self.MammographyRoI.key])

        config.API.LOG('RoI segmentation...')
        self.MammographyRoI.infer()
        self.process_unify()
        self.correct()
        self.process([self.DensityEstimation.key])

        config.API.LOG('Density estimation...')
        self.DensityEstimation.infer()
        self.clear_cropped()

        config.API.LOG('Mass segmentation...')
        self.process([self.MassSegmentation.key])
        self.MassSegmentation.infer()
        self.response()

    def check_status(self):
        for _ in range(config.API.MAX_QUEUE_LENGTH):
            try:
                data = self.mp_queue.get(
                    block=True, timeout=config.API.TTL)
            except eq.Empty:
                break
            self.stack(data)

    def stack(self, data):
        print(data['message'])
        data = [
            {
                'side': k,
                'image': load_image(v),
                'channel': data['channel'],
                #TODO: rewrite this
                'thresholds': { 
                    k: data['data']['thresholds'][k] if k in data['data']['thresholds'] else .5 
                    for k in [
                        'radiant_node',
                        'intramammary_lymph_node',
                        'calcification',
                        'mask',
                        'structure',
                        'border',
                        'shape',
                        'calcification_malignant',
                        'local_structure_perturbation',
                    ]
                }
            }
            for k, v in data['message'].items()
        ]
        self.queue.extend(data)

    def clear(self):
        self.queue.clear()
        for k in config.API.KEYS:
            self[k].clear()
        gc.collect()

    def correct(self):
        config.API.LOG('Cropping to RoI...')
        self.crop_to_roi(self.MammographyRoI.predictions)
        for k in config.API.KEYS:
            self[k].queue.clear()

    def process(self, allowed_keys):
        # with mp.Pool(config.WORKERS_NB) as pool:
        #    pool.map(self._subprocess, self.queue)
        for data in self.queue:
            self._subprocess(data, allowed_keys)

    def _subprocess(self, data, allowed_keys=[]):
        for p, keys in config.PROCESS.MAP.items():
            if not any([ k in allowed_keys for k in keys ]):
                continue
            processed = p.process(data, self)
            for k in keys:
                if 'cimage' in processed.keys():
                    processed['image'] = processed['cimage']
                self[k].append(processed)

    def process_unify(self):
        for k, el in enumerate(self.queue):
            if el['image'].dtype == np.uint16:
                mask = self.MammographyRoI.predictions['{}|{}'.format(el['channel'], el['side'])]
                el['image'] = ps.convert_and_unify(el['image'], mask['whole'] > .1)

    @staticmethod
    def crop(data, masks):
        key = config.API.PID_SIDE2KEY(data['channel'], data['side'])
        mask = masks[key]['whole']
        coeff = np.array(data['image'].shape[:2]) / np.array(mask.shape[:2])
        coeff = np.expand_dims(coeff, -1)

        coords = np.array(np.where(mask > config.PROCESS.RoI.THRESHOLD))
        coords = (coords * coeff).astype(np.int)
        if coords.shape[1] == 0:
            config.API.LOG(
                'Image with key=={} has no RoI segmentation'.format(key))
            return data

        y_max, x_max = coords.max(1)
        y_min, x_min = coords.min(1)

        data['cimage'] = data['image'][y_min: y_max, x_min: x_max]
        data['yxmin_yxmax'] = (y_min, x_min, y_max, x_max)
        return data

    def crop_to_roi(self, masks):
        for i, data in enumerate(self.queue):
            self.queue[i] = self.crop(data, masks)

    def clear_cropped(self):
        for i, data in enumerate(self.queue):
            data.pop('cimage')
    
    def response(self):
        channels = { q['channel']: q['thresholds'] for q in self.queue }
        keys = set(
            config.API.PID_SIDE2KEY(q['channel'], q['side']) 
            for q in self.queue )

        predictions = dict()
        for k in keys:
            predictions[k] = self.DensityEstimation.predictions[k]
            predictions[k].update(self.MassSegmentation.predictions[k])
            predictions[k].update(self.MammographyRoI.predictions[k])

        for c in channels.keys():
            channel = { 
                k.split('|')[-1]: v
                for k, v in predictions.items() 
                if '|'.join(k.split('|')[:-1]) == c
            }
            for side in channel.keys():
                channel[side]['original'] = [
                    q['image'] for q in self.queue 
                    if q['side'] == side and q['channel'] == c
                ].pop()
            response = rs.build_response(channel, c, self, thresholds=channels[c])
            self.r_api.publish(c.split('.')[-1], response)


class LQueue:
    def __init__(self, qm, key):
        self.qm = qm
        self.key = key
        self.learner = config.SHARED.models[key]
        self.queue = list()
        self.predictions = dict()
        self.dataset = ds.InferenceDataset(
            self, config.MAMMOGRAPHY_PARAMS.MODELS[key]['transform'])
        self.datagen = torch.utils.data.DataLoader(
            self.dataset, batch_size=1,
            num_workers=config.WORKERS_NB,
            collate_fn=ds.inference_collater if key == 'MammographyRoI' else None)

    def append(self, value):
        self.queue.insert(0, value)

    def clear(self):
        self.predictions.clear()
        self.queue.clear()

    def infer(self):
        self.predictions = self.learner.validate(self.datagen)

    def __getitem__(self, idx):
        return self.queue[idx]

    def __len__(self):
        return len(self.queue)
