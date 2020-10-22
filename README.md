# Simple dockerizable flask app using REST API

## Build
    docker build -t new_image .

## Run
    docker run -it --rm -d -p 8080:80 --name flask_docker_example new_image