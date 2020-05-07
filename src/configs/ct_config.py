import easydict

from src.models.unet import TernausNet


PARAMS = easydict.EasyDict()
PARAMS.MEAN = [90.2751662940309]
PARAMS.STD = [60.05609845525738]

MODELS = {
    'AlbuNet/RetinaNet': {
        'model': TernausNet.AlbuNet,
        'kwargs': {
            'num_classes': 1,
            'is_deconv': False
        },
        'RetinaNet': {},
        'path': 'latest.pkl',
    }
}
