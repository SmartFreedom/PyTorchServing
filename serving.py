from flask import Flask, request, jsonify

from src.models.models import Models, Preprocess
from src.configs import config
import utils

import torch
import time
import os

utils.make_dirs()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
torch.cuda.set_device(config.CUDA_IDX)

models = Models()
preprocess = Preprocess()


# set flask params
app = Flask(__name__)


@app.route("/")
def information():
    return "PyTorch Serving\n"


@app.route('/predict', methods=['GET'])
def predict():
    model_name = request.args['name']
    input_path = request.args['inpt']
    app.logger.info("Classifying image %s" % (model_name))
    t = time.time()  # get execution time

    model = models[model_name]
    datagen = preprocess(model_name, input_path)
    results = model.validate(datagen)
    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))

    return jsonify(results.__str__())

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=config.API.PORT)
