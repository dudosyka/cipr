from django.shortcuts import render
from django.http import HttpResponse
import builtins
import json

import os
import datetime

import tensorflow as tf
import numpy as np

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def printTime():
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))


class Prediction:
    def __init__(self, model):
        self.model = model

    def predict(self, sample_text):
        res = []
        for item in os.scandir('./models/fts'):
            i = 0
            if item.is_dir():
                self.model.load_weights('/home/dudosyka/Documents/HackITMO/neuro/models/fts/' + item.name)
                predictions = self.model.predict(np.array([sample_text]))
                res.append({"code": item.name, "prediction": str(predictions[0][0]) })

        return res


def index(request):
    prediction = Prediction(builtins.modelProto)
    predictions = prediction.predict(request.GET['req'])
    return HttpResponse(json.dumps(predictions))

# Create your views here.
