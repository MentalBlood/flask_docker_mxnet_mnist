from __future__ import print_function
import json
import base64
import argparse

parser = argparse.ArgumentParser(description='Generate JSON for request')
parser.add_argument('files', metavar='N', type=str, nargs='+',
                    help='paths to files to send in request')
args = parser.parse_args()

imgPaths = args.files
def readImage(path):
	result = None
	with open(path, 'rb') as image:
		imageBytes = open(path, 'rb').read()
		result = base64.encodebytes(imageBytes).decode("utf-8")
	return result
img = list(map(readImage, imgPaths))
jsonString = json.dumps({'images': img})
print(jsonString)