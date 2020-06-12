from tqdm import tqdm
from IPython.display import clear_output

import src.modules.dataset as ds
import src.modules.augmentations as augs
import src.utils.visualisation as vs


class Inference:
    def __init__(self, queue, verbose=False):
        self.verbose = verbose
        self.queue = queue

    def __call__(self):
        self.queue.model.model.eval()
        for i, data in tqdm(enumerate(self.queue.datagen)):
            predict = ds.infer_on_batch(self.queue.model.model, data)
            predict = augs._rotate_mirror_undo(predict)
            shape = data['shape'][0]
            image = data['image'][..., :shape[0], :shape[1]]
            predict = predict[..., :shape[0], :shape[1]]
            self.queue.predictions.append({
                'pid': data['pid'][0],
                'side': data['side'][0],
                'predict': predict
            })

            if self.verbose:
                clear_output(wait=True)
                vs.show_segmentations(image[0, 0].numpy(), predict)
