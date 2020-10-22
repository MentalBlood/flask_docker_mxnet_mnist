from __future__ import print_function

from flask import Flask, request
from flask_restful import Resource, Api
from logging.config import dictConfig

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as F

def buildNet(net):
	with net.name_scope():
		# First convolution
		net.add(nn.Conv2D(10, kernel_size=5, activation='tanh'))
		net.add(nn.MaxPool2D(pool_size=2, strides=2))
		# Second convolution
		net.add(nn.Conv2D(25, kernel_size=5, activation='tanh'))
		net.add(nn.MaxPool2D(pool_size=2, strides=2))
		# First fully connected layers with 20 neurons
		net.add(nn.Dense(20, activation='tanh'))
		# Second fully connected layer with as many neurons as the number of classes
		net.add(nn.Dense(10, activation='tanh'))
		return net

net = buildNet(nn.Sequential())
net.load_parameters('net.params')

import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt
import numpy as np

logBuffer = ''

def verify_loaded_model(net):
	global logBuffer

	"""Run inference using ten random images.
	Print both input and output of the model"""

	def transform(data, label):
		return data.astype(np.float32)/255, label.astype(np.float32)

	imagesNumber = 10
	# Load random images from the test dataset
	sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False,
																	  transform=transform),
										   imagesNumber, shuffle=True)

	for data, label in sample_data:

		# Display the images
		#img = nd.transpose(data, (1, 0, 2, 3))
		#img = nd.reshape(img, (28, imagesNumber*28, 1))
		#imtiles = nd.tile(img, (1, 1, 3))
		#plt.imshow(imtiles.asnumpy())
		#plt.show()

		# Display the predictions
		data = nd.transpose(data, (0, 3, 1, 2))
		out = net(data)
		predictions = nd.argmax(out, axis=1)
		logBuffer += 'Model predictions: ' + str([int(p) for p in predictions.asnumpy()]) + '\n'

		break

app = Flask(__name__)
api = Api(app)

@app.route('/log', methods=['POST'])
def json_example():
	#args = request.get_json()
	verify_loaded_model(net)
	return {'log': logBuffer}

if __name__ == '__main__':
	app.run()