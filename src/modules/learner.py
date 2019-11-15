import torch
from tqdm import tqdm


def get_model(model, checkpoint=None, map_location=None, devices=None):
    model.cuda()

    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location) #.module.state_dict()
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd}
        print('Overlapped keys: {}'.format(len(sd.keys())))
        msd.update(sd)
        model.load_state_dict(msd)

    if devices is not None:
        model = torch.nn.DataParallel(model, device_ids=devices)

    return model


def to_single_channel(model):
    conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    nstd = dict()
    for key, param in model.conv1[0].state_dict().items():
        nstd[key] = param.sum(1).unsqueeze(1)
        print('Summed over: {}'.format(key))
    conv1.load_state_dict(nstd)
    model.conv1[0] = conv1
    return model


def freeze(model, unfreeze=False):
    children = list(model.children())
    if hasattr(model, 'children') and len(children):
        for child in children:
            freeze(child, unfreeze)
    elif hasattr(model, 'parameters'):
        for param in model.parameters():
            param.requires_grad = unfreeze
            
            
def unfreeze_bn(model):
    predicat = isinstance(model, torch.nn.BatchNorm2d)
    predicat |= isinstance(model, bn.ABN)
    predicat |= isinstance(model, bn.InPlaceABN)
    predicat |= isinstance(model, bn.InPlaceABNSync)
    if predicat:
        for param in model.parameters():
            param.requires_grad = True

    children = list(model.children())
    if len(children):
        for child in children:
            unfreeze_bn(child)
    return None


class Inference:
    def __init__(self, model):
        self.model = model

    def make_step(self, data, training=False):
        image = self._format_input(data)
        prediction = self.model(image)
        results = self._format_output(prediction, data)
        image = image.data.cpu()

        return results

    def validate(self, datagen):
        torch.cuda.empty_cache()
        self.model.eval()
        meters = list()
        with torch.no_grad():
            for data in tqdm(datagen.dataset):
                meters.append(self.make_step(data, training=False))
        return meters

    @staticmethod
    def _format_input(data):
        image = torch.autograd.Variable(data['image']).cuda().float()

        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)

        return image

    @staticmethod
    def _format_output(prediction, data):
        results = { k: v.data.cpu() if torch.is_tensor(v) else v for k, v in data.items() if k != 'image' }
        results.update({ 'prediction': prediction.data.cpu() if torch.is_tensor(prediction) else prediction })
        return results
