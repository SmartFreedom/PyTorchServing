import redis
import time
import traceback
import imageio
import json

from src.api import queue_manager as qm
from src.configs import config


class RedisAPI:
    def __init__(self, manager: qm.QueueManager):
        self.manager = manager
        self.jd = json.JSONDecoder()
        self.r_connector = redis.StrictRedis(
            host=config.API.REDIS.HOST,
            port=config.API.REDIS.PORT,
            db=config.API.REDIS.DB
        )
        self.subscriber = self.r_connector.pubsub()
        self.subscriber.psubscribe(config.API.REDIS.I_CHANNEL)

    def check(self):
        try:
            pause = True
            while pause:
                print("Waiting For redisStarter...")
                message = self.subscriber.get_message()
                if message:
                    command = message['data']
                    if command == config.API.REDIS.START:
                        pause = False
                time.sleep(1)
            print("Permission to start...")

        except Exception as e:
            print("EXCEPTION")
            print(str(e))
            print(traceback.format_exc())

    def listen(self):
        while True:

            time.sleep(1)
#             if len(self.queue) >= config.MAX_QUEUE_LENGTH:
#                 continue

            message = self.subscriber.get_message()
            if message:
                print(message)
                channel = message['channel']
                message = self.jd.decode(message['data'].decode())
                self.manager.process(channel, data={
                    k: imageio.imread(v)
                    for k, v in message['urls'].items()
                })
