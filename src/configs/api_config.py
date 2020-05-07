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
API.REDIS.CHANNEL = 'requests.mammography_screening.*'
API.REDIS.START = 1
