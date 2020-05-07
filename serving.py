from src.configs import config
from src.api import flask
from src.api import redis

if __name__ == '__main__':
    config.SHARED.INIT()
    redis.redis_check()
    flask.app.run(host="0.0.0.0", debug=config.API.DEBUG, port=config.API.PORT)
