#!/bin/bash
#nohup redis-server --port 9358 > $HOME/redis.log 2>&1 &
# to renew the prompt & init conda setup
/bin/bash
conda activate crt
python serving.py > $HOME/torch.log 2>&1 &

