from collections import defaultdict
import multiprocessing as mp
import queue as eq
import easydict
import imageio
import torch

from src.configs import config
import src.modules.dataset  as ds


class QueueManager(easydict.EasyDict):
    def __init__(self, mp_queue: mp.Queue):
        super(QueueManager, self).__init__()
        self.mp_queue = mp_queue
        self.keys = config.API.KEYS
        self.update({k: LQueue(self, k) for k in self.keys})

    def check_status(self):
        for _ in range(config.API.MAX_QUEUE_LENGTH):
            try:
                data = self.mp_queue.get(
                    block=True, timeout=config.API.TTL)
            except eq.Empty:
                break
            self.process(data['channel'], data={
                k: imageio.imread(v)
                for k, v in data['message'].items()
            })

    def process(self, channel, data):
        for p, keys in config.PROCESS_MAP.items():
            processed = p.process(data)
            for k in keys:
                for side, el in processed.items():
                    self[k].append({
                        'channel': channel,
                        'side': side,
                        'image': el
                    })


class LQueue:
    def __init__(self, qm, key):
        self.qm = qm
        self.model = config.SHARED.models[key]
        self.queue = list()
        self.predictions = list()
        self.dataset = ds.InferenceDataset(self)

        self.datagen = torch.utils.data.DataLoader(
            self.dataset, batch_size=1,
            num_workers=config.WORKERS_NB,
            collate_fn=ds.inference_collater)

    def append(self, value):
        self.queue.insert(0, value)

    def __getitem__(self, idx):
        return self.queue[idx]

    def __len__(self):
        return len(self.queue)
