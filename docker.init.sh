#!/bin/bash

usage() { echo "Usage: $0 -i GPUINDEX -v REDIS_DB_V -d DATA_VOLUME" 1>&2; exit 1; }

while getopts ":i:v:d:" o; do
    case "${o}" in
        i) i=${OPTARG};;
        v) v=${OPTARG};;
        d) d=${OPTARG};;
        *) usage;;
    esac
done
shift $((OPTIND-1))

if [ -z "${i}" ] || [ -z "${v}" ] || [ -z "${d}" ]; then
    usage
fi

echo "CUDA visible device set to ... $i"
echo "REDIS database version set to ... $v"
echo "Local data volume set to ... $d"

export TORCH_SERVER_V=0.4
nvidia-docker image build -t torch-server:$TORCH_SERVER_V . --build-arg REDIS_DB_V=$v
NV_GPU=$i nvidia-docker run --rm -it -p 0.0.0.0:9767:22/tcp -p 0.0.0.0:9769:9769/tcp --name torchy  torch-server:$TORCH_SERVER_V --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=$i -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $d:/opt/entrypoint/dataset
