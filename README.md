# PyTorchServing
PyTorch Serving

## Requirements
 - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
 - [git LFS](https://git-lfs.github.com/)

## Launching

Build & run docker:  

```bash
git clone
cd PyTorchServing
git lfs install
git fetch
git merge/origin
./docker.init.sh -i $GPU_INDEX
```  

Inside docker:  
```bash
(crt) entrypoint$ python serving.py 
CTRL+P+Q
```
