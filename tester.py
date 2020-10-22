from __future__ import print_function
import requests
import json

addr = 'http://localhost:5000'
test_url = addr + '/log'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# send http request with image and receive response
img_path = '3.png'
img = open(img_path, 'rb').read()
response = requests.post(test_url, data=img, headers=headers)
# decode response
print(json.loads(response.text))