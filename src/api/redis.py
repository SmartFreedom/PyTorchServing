import redis
import time
import traceback
import imageio
import json

from src.configs import config


class RedisAPI:
    def __init__(self, mp_queue):
        """mp_queue: mp.Queue"""
        self.mp_queue = mp_queue
        self.jd = json.JSONDecoder()
        self.je = json.JSONEncoder()
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
            message = self.subscriber.get_message(config.API.TTL)
            if message:
                self.mp_queue.put({
                    'channel': message['channel'].decode(),
                    'message': self.jd.decode(message['data'].decode())['urls'],
                    'data': self.jd.decode(message['data'].decode()),
                })

    def publish(self, case_id, response):
        self.r_connector.publish(
            channel=config.API.REDIS.O_CHANNEL.format(case_id=case_id),
            message=self.je.encode(response)
        )
