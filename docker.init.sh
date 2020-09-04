#!/bin/bash

usage() { echo "Usage: $0 -i GPUINDEX" 1>&2; exit 1; }

while getopts ":i:" o; do
    case "${o}" in
        i) i=${OPTARG};;
        *) usage;;
    esac
done
shift $((OPTIND-1))

if [ -z "${i}" ]; then
    usage
fi

echo "CUDA visible device set to ... $i"
export TORCH_SERVER_V=0.4
nvidia-docker image build -t torch-server:$TORCH_SERVER_V .
NV_GPU=$i nvidia-docker run --rm -it -p 0.0.0.0:9767:22/tcp -p 0.0.0.0:9769:9769/tcp --name torchy  torch-server:$TORCH_SERVER_V --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=$i -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
