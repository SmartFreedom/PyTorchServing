#!/bin/bash
nvidia-docker image build -t torch-server:0.2 .
nvidia-docker run --rm -it -p 0.0.0.0:9767:22/tcp -p 0.0.0.0:9769:9769/tcp --name torchy  torch-server:0.2 --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
