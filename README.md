# Simple dockerizable mxnet MNIST/FashionMNIST prediction app using Flask-powered REST API

## Build
    docker build -t new_image .

## Run
    docker run -it --rm -d -p 8080:80 --name flask_docker_example new_image