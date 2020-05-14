import torch

from src.api import queue_manager as qm
from src.api import flask
from src.api import redis
from src.configs import config
from src.modules import dataset as ds


if __name__ == '__main__':
    config.SHARED.INIT()
    manager = qm.QueueManager(keys=['MammographyRoI', 'MammographyDencity'])
    idataset = ds.InferenceDataset(manager['MammographyRoI'])

    datagen = torch.utils.data.DataLoader(
        idataset, batch_size=1,
        num_workers=config.WORKERS_NB,
        collate_fn=ds.inference_collater)

    r_api = redis.RedisAPI(manager)
    r_api.check()
    r_api.listen()

    # flask.app.run(host="0.0.0.0", debug=config.API.DEBUG, port=config.API.PORT)
