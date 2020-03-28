#!/bin/bash
nohup redis-server --port 9358 > $HOME/redis.log 2>&1 &
nohup python serving.py > $HOME/torch.log 2>&1 &
