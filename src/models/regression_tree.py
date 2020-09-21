import numpy as np
import pickle
from collections import OrderedDict, defaultdict
import sklearn.tree
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import xgboost


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


def extract_features(v):
    sides = ['r', 'l']
    attributes = [
        'distortions', 'lymph_node', 
        'calcifications_benign', 'calcifications_malignant' ]

    mass_attributes = [
        'inhomogen', 'obscure_margin', 
        'irregular_shape', 'with_calc' ]

    data = {}

    for side in sides:
        for name in attributes:
            data['{}|{}'.format(side, name)] = v['prediction'][side][name]['response']['yes']
        for name in mass_attributes:
            data['{}|{}'.format(side, name)] = v['prediction'][side]['mass']['response'][name]

    for name in attributes:
        data['A{}|{}'.format(side, name)] = abs(
            v['prediction']['r'][name]['response']['yes']
            - v['prediction']['l'][name]['response']['yes']
        )
    for name in mass_attributes:
        data['A{}|{}'.format(side, name)] = abs(
            v['prediction']['r']['mass']['response'][name]
            - v['prediction']['l']['mass']['response'][name]
        )

    findings = defaultdict(list)
    for el in v['findings']:
        findings[el['type']].append(el['geometry'])

    for name in attributes + ['mass']:
        # data['area|sum|{}'.format(name)] = sum(findings[name])
        data['amount|{}'.format(name)] = len(findings[name])
        # data['area|max|{}'.format(name)] = max(findings[name] + [0])

    data['density'] = v['prediction']['density']['response']['ab']
    return data


class ProbabilityClassifier():

    def __init__(self, checkpoint):
        self.classifier = xgboost.XGBClassifier(random_state=0)
        self.classifier.load_model(checkpoint)

    def __call__(self, response):
        feature_vector = pd.DataFrame([extract_features(response)])
        pred = self.classifier.predict_proba(feature_vector)
        return pred[0, 1] # 0th sample & prob of 1st class
