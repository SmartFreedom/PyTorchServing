from src.configs import config
from src.api import flask


if __name__ == '__main__':
    config.SHARED.INIT()
    flask.app.run(host="0.0.0.0", debug=config.API.DEBUG, port=config.API.PORT)
