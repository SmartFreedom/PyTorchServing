import numpy as np
import pickle
from collections import OrderedDict
import sklearn.tree
from sklearn.tree import DecisionTreeRegressor


MASK_NAMES = {
    'fpn': {
        0: 'calcification',
        1: 'mask',
    },
    'head': {
        0: 'structure',
        1: 'border',
        2: 'shape',
        3: 'calcification_malignant',
        4: 'local_structure_perturbation',
    }
}


class ProbabilityRegressor():
    
    feature_from_mask = OrderedDict([('mass', ['fpn', 'mask']), 
                ('mass_irregular_shape', ['head', 'shape']),  
                ('mass_spiculate_margin', ['head', 'border']), 
                ('mass_inhomogen', ['head', 'structure']), 
                ('calcifications_benign', ['fpn', 'calcification']), 
                ('calcifications_malignant', ['head', 'calcification_malignant']), 
                ('local_perturbation', ['head', 'local_structure_perturbation'])])

    def __init__(self, checkpoint):
        with open(checkpoint, 'rb') as fid:
            self.regressor = pickle.load(fid)

    def __call__(self, masks):
        feature_vector = np.array([
            np.max(masks[v[0]][v[1]]) 
            for k, v in self.feature_from_mask.items()])
        pred = self.regressor.predict(feature_vector.reshape(1, -1))
        return pred[0]
