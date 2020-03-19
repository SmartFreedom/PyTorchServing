import easydict

PARAMS = easydict.EasyDict()
PARAMS.MEAN = [74.77618355029848]
PARAMS.STD = [31.738553261533994]

PARAMS.CROP_SIDE = 1024
PARAMS.CNN_SIDE = 512
PARAMS.CROP_STEP = 256
PARAMS.SIDE = 512
PARAMS.DROPOUT = .2

PARAMS.MIN_DIAMETER = 15
PARAMS.MAX_DIAMETER = 108

PARAMS.TARGET_NAME = 'label'

PARAMS.ASSYMETRY_TYPES = [
    "Нет",
    "Есть Локальная (очаговая)",
    "Есть Тотальная",
    "Есть Локальная с кальцинатами",
    "Есть Динамическая"
]

PARAMS.DENSITY_TYPES = [
    'A',
    'B',
    'C',
    'D'
]
