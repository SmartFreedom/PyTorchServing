import easydict


# set some deployment settings
API = easydict.EasyDict()
API.ROOT = 'https://label.cmai.tech'
API.CASES = API.ROOT + '/api/v1/cases'
API.KEY = 'jMTCJiJNETMDpwystkl25dFgPbDVpmiSl0Cx6k5pZ7xcUNKu4hbLOpo2UWgIOq8ZBZ7U5Q1djTsyPdmoekNAU3RqhP2kMhp8A5Ef80YDLIchZOGNi77rUrsdlTatwEva'
API.PORT = 9769
API.DEBUG = True

API.REDIS = easydict.EasyDict()
API.REDIS.HOST = '10.20.12.13'
API.REDIS.PORT = 6379
API.REDIS.DB = 3
API.REDIS.I_CHANNEL = 'requests.xray_mammography.*'
API.REDIS.O_CHANNEL = 'analyse_result.{case_id}'
API.REDIS.START = 1
API.KEYS = [ 'MammographyRoI', 'DensityEstimation', 'AsymmetryEstimation', 'MassSegmentation' ]
API.PID_SIDE2KEY = lambda pid, side: '{}|{}'.format(pid, side)
API.LOG = print

API.MAX_QUEUE_LENGTH = 1
API.TTL = 1
