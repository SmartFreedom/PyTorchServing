import flask
from flask import Flask, request, jsonify

import numpy as np
import time
import io

from src.configs import config


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
    app.logger.info("Classifying image %s" % model_name)
    t = time.time()  # get execution time

    model = config.SHARED.models[model_name]
    datagen = config.SHARED.preprocess(model_name, input_path)
    results = model.validate(datagen)
    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % dt)
    buffer = io.BytesIO()  # create buffer
    np.savez_compressed(buffer, **results)

    buffer.seek(0)  # This simulates closing the file and re-opening it.
    #  Otherwise the cursor will already be at the end of the
    #  file when flask tries to read the contents, and it will
    #  think the file is empty.

    return flask.send_file(
        buffer,
        # as_attachment=True,
        attachment_filename='results.npz',
        mimetype='mask/npz'
    )
