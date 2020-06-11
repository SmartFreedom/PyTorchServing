import torch
import easydict

from src.api import queue_manager as qm
from src.api import flask
from src.api import redis
from src.configs import config
from src.modules import dataset as ds
import src.modules.learner as lrn
import src.utils.preprocess as ps
from src.modules import inference

import multiprocessing as mp


if __name__ == '__main__':
    mp_queue = mp.Queue()
    r_api = redis.RedisAPI(mp_queue)
    r_api.check()

    redis_process = mp.Process(target=r_api.listen)
    redis_process.start()

    config.SHARED.INIT()

    manager = qm.QueueManager()

    manager.start(mp_queue=mp_queue)

    # flask.app.run(host="0.0.0.0", debug=config.API.DEBUG, port=config.API.PORT)
