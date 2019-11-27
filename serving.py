import flask
from flask import Flask, request, jsonify

from src.models.models import Models, Preprocess
from src.configs import config
import utils

import torch
import time
import os

import numpy as np
import io

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


@app.route('/predict', methods=['POST'])
def predict():
    print(request.data)

    model_name = request.json['name']
    input_path = request.json['inpt']
    app.logger.info("Classifying image %s" % (model_name))
    t = time.time()  # get execution time

    model = models[model_name]
    datagen = preprocess(model_name, input_path)
    results = model.validate(datagen)
    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))

    buffer = io.BytesIO()  # create buffer
    np.savez_compressed(buffer, **results)

    buffer.seek(0)  # This simulates closing the file and re-opening it.
    #  Otherwise the cursor will already be at the end of the
    #  file when flask tries to read the contents, and it will
    #  think the file is empty.

    return flask.send_file(
        buffer,
        #as_attachment=True,
        attachment_filename='results.npz',
        mimetype='mask/npz'
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=config.API.PORT)
