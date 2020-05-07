from flask import Flask, request, jsonify
from src.configs import config
from src.api import flask

# set flask params


if __name__ == '__main__':
    flask.app.run(host="0.0.0.0", debug=config.API.DEBUG, port=config.API.PORT)
