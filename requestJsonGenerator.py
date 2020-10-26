import json
import base64
import argparse

parser = argparse.ArgumentParser(description='Generate JSON for request')
parser.add_argument('--dataset', type=str, nargs='?',
					choices=['MNIST', 'FashionMNIST'],
					default='MNIST',
                    help='dataset model should be trained on')
parser.add_argument('--images', metavar='image', type=str, nargs='+',
                    help='paths to files to send in request')
args = parser.parse_args()

imgPaths = args.images
netName = args.dataset
def readImage(path):
	result = None
	with open(path, 'rb') as image:
		imageBytes = image.read()
		result = base64.encodebytes(imageBytes).decode("utf-8")
	return result
img = list(map(readImage, imgPaths))
jsonString = json.dumps({'images': img, 'netName': netName})
print(jsonString)