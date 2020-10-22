from __future__ import print_function

from flask import Flask, request
from flask_restful import Resource, Api
from logging.config import dictConfig

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as F

import cv2

nets = {
	'MNIST': {
		'layers': [
			nn.Conv2D(10, kernel_size=5, activation='tanh'),
			nn.MaxPool2D(pool_size=2, strides=2),
			nn.Conv2D(25, kernel_size=5, activation='tanh'),
			nn.MaxPool2D(pool_size=2, strides=2),
			nn.Dense(20, activation='tanh'),
			nn.Dense(10, activation='tanh')
		],
		'paramsFile': 'net.params'
	}
}

def buildNet(netName):
	net = nn.Sequential()
	for layer in nets[netName]['layers']:
		net.add(layer)
	paramsFile = nets[netName]['paramsFile']
	if paramsFile:
		net.load_parameters(paramsFile)
	return net

mnistNet = buildNet('MNIST')

import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt
import numpy as np

def verifyLoadedModel(net, data):
	data = nd.transpose(nd.array(data), (0, 3, 1, 2))
	out = net(data)
	predictions = nd.argmax(out, axis=1)

	return {'predictions': [int(p) for p in predictions.asnumpy()]}

app = Flask(__name__)
api = Api(app)

def prepareImage(img):
	nparr = np.frombuffer(img, np.uint8)
	grayscale = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
	return np.array(grayscale.astype(np.float32)/255).reshape(28, 28, 1)

def prepareData(data):
	return list(map(prepareImage, data))

@app.route('/log', methods=['POST'])
def json_example():
	data = request.data.split(b'end')
	return verifyLoadedModel(mnistNet, prepareData(data))

if __name__ == '__main__':
	app.run()