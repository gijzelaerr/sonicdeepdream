#!/bin/bash

#for ubuntu 16.04

sudo apt-get -y install libfreetype6-dev  python-virtualenv libpython3-dev nvidia-cuda-dev libffi-dev
virtualenv -p python .virtualenv

.virtualenv/bin/pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl 
.virtualenv/bin/pip install -r requirements.txt
