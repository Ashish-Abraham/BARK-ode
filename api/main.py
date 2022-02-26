import tensorflow as tf
from flask import Flask, render_template, url_for, request, redirect, jsonify
from PIL import Image
import numpy as np
import os
from torch.functional import split
from ml_predict import predict_dog
import base64
from decouple import config

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    key_dict = request.get_json()
    image = key_dict["image"]
    imgdata = base64.b64decode(image)
    dog = predict_breed(imgdata)
    response = {
        "result": dog,
    }
    response = jsonify(response)
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
