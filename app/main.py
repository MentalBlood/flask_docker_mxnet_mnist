from flask import Flask, request
from flask_restful import Resource, Api

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as F

import cv2
import base64

nets = {
	'MNIST': {
		'modelFile': 'mnist-symbol.json',
		'paramsFile': 'mnist.params',
		'textLabels': None
	},
	'FashionMNIST': {
		'modelFile': 'fashion-mnist-symbol.json',
		'paramsFile': 'fashion-mnist.params',
		'textLabels':	[
							't-shirt',
							'trouser',
							'pullover',
							'dress',
							'coat',
							'sandal',
							'shirt',
							'sneaker',
							'bag',
							'ankle boot'
						]
	}
}

def buildNets():
	for netName in nets:
		model = nets[netName]['modelFile']
		parameters = nets[netName]['paramsFile']
		nets[netName]['net'] = nn.SymbolBlock.imports(model, ['data'], parameters)

buildNets()

import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms

import numpy as np

def verifyLoadedModel(net, data, textLabels):
	data = nd.transpose(nd.array(data), (0, 3, 1, 2))
	out = net(data)
	predictions = nd.argmax(out, axis=1).asnumpy()

	if textLabels:
		return {'predictions': [(int(p), textLabels[int(p)]) for p in predictions]}
	else:
		return {'predictions': [int(p) for p in predictions]}

app = Flask(__name__)
api = Api(app)

def prepareImage(imageString):
	imageBytes = base64.decodebytes(imageString.encode('utf-8'))
	nparr = np.frombuffer(imageBytes, np.uint8)
	grayscale = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
	return np.array(grayscale.astype(np.float32)/255).reshape(28, 28, 1)

def prepareData(data):
	return list(map(prepareImage, data))

@app.route('/classify', methods=['POST'])
def json_example():
	netName = request.get_json()['dataset']
	net = nets[netName]['net']
	data = request.get_json()['images']
	textLabels = nets[netName]['textLabels']
	return verifyLoadedModel(net, prepareData(data), textLabels)

if __name__ == '__main__':
	app.run()