from collections import defaultdict

from src.configs import config


class QueueManager(dict):
    def __init__(self, keys: list):
        super(QueueManager, self).__init__()
        self.update({ k: self.LQueue(self, k) for k in keys })

    def check_status(self):
        # First check in local queue, then in unprocessed redis data
        for lq in self.values():
            if len(lq.queue) == 0:
                #TODO: implement in parallel
                break


    def process(self, channel, data):
        for p, keys in config.PROCESS.MAP.items():
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
            self.key = key
            self.qm = qm
            self.queue = list()
            self.responses = list()

        def pop(self):
            if not len(self.queue):
                self.qm.check_status()
            return self.queue.pop()

        def append(self, value):
            self.queue.insert(0, value)
