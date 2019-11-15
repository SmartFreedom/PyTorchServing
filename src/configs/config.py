from pathlib import Path
import os
import easydict

MEAN = [90.2751662940309]
STD = [60.05609845525738]

DEVICES = [0]

PATHS = easydict.EasyDict()
PATHS.DATA_ROOT = Path(os.getcwd())/'data'
PATHS.MODELS = PATHS.DATA_ROOT/'models'
PATHS.RESULTS = PATHS.DATA_ROOT/'results'
PATHS.LOGDIR = PATHS.DATA_ROOT/'logdir'


# set the URL where you can download your model weights
MODELS = {
    'AlbuNet': 'latest.pkl'
}

BATCH_SIZE = 8

CUDA_VISIBLE_DEVICES = "1"
CUDA_IDX = 0


# set some deployment settings
API = easydict.EasyDict()
API.ROOT = 'https://label.cmai.tech'
API.CASES = API.ROOT + '/api/v1/cases'
API.KEY = 'jMTCJiJNETMDpwystkl25dFgPbDVpmiSl0Cx6k5pZ7xcUNKu4hbLOpo2UWgIOq8ZBZ7U5Q1djTsyPdmoekNAU3RqhP2kMhp8A5Ef80YDLIchZOGNi77rUrsdlTatwEva'
API.PORT = 9989
