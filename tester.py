from __future__ import print_function
import requests
import json
import numpy as np

testUrl = 'http://localhost:5000/log'
headers = {'content-type': 'application/json'}

imgPaths = ['3.png', '1.png']
def readImage(path):
	result = open(path, 'rb').read()
	return result
img = list(map(readImage, imgPaths))
response = requests.post(testUrl, data=b'end'.join(img), headers=headers)

print(json.loads(response.text))