import redis
import time
import traceback

from src.configs import config


def redis_check():
    try:
        r = redis.StrictRedis(host=config.API.REDIS.HOST, port=config.API.REDIS.PORT)
        p = r.pubsub()
        p.psubscribe(config.API.REDIS.CHANNEL)
        PAUSE = True

        while PAUSE:
            print("Waiting For redisStarter...")
            message = p.get_message()
            if message:
                command = message['data']
                if command == config.API.REDIS.START:
                    PAUSE = False
            time.sleep(1)
        print("Permission to start...")

    except Exception as e:
        print("EXCEPTION")
        print(str(e))
        print(traceback.format_exc())

