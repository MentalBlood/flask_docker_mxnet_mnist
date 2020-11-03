FROM tiangolo/uwsgi-nginx-flask:python3.8

COPY ./app /app

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt