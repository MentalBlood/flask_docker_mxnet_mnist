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

def verify_loaded_model(net, data):
	"""Run inference using ten random images.
	Print both input and output of the model"""

	def transform(data, label):
		return data.astype(np.float32)/255, label.astype(np.float32)

	imagesNumber = 1
	# Load random images from the test dataset
	#sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False,
	#																  transform=transform),
	#									   imagesNumber, shuffle=True)



	#for data, label in sample_data:

		# Display the images
		#img = nd.transpose(data, (1, 0, 2, 3))
		#img = nd.reshape(img, (28, imagesNumber*28, 1))
		#imtiles = nd.tile(img, (1, 1, 3))
		##images = imtiles.asnumpy()
		#plt.imshow(imtiles.asnumpy())
		#plt.show()

		# Display the predictions
	rgb_weights = [0.2989, 0.5870, 0.1140]
	grayscale_image = np.dot(data[...,:3], rgb_weights)
	data = np.array([grayscale_image.astype(np.float32)/255]).reshape(1, 28, 28, 1)
	print('SHAPE IS', data.shape)
	data = nd.transpose(nd.array(data), (0, 3, 1, 2))
	out = net(data)
	predictions = nd.argmax(out, axis=1)

	return {'predictions': [int(p) for p in predictions.asnumpy()]}

app = Flask(__name__)
api = Api(app)

@app.route('/log', methods=['POST'])
def json_example():
	r = request
	# convert string of image data to uint8
	nparr = np.fromstring(r.data, np.uint8)
	# decode image
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return verify_loaded_model(net, img)
	response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
	return response

if __name__ == '__main__':
	app.run()