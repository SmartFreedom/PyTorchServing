import os
import cv2
import scipy
import skimage
import addict
import numpy as np

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from src.utils import rle
from src.configs import config
from src.models import regression_tree as rt


def build_rle_findings(pred, threshold, lower_bound, key, side):
    response = list()

    mask = pred > threshold
    labeled, colours = scipy.ndimage.label(mask)
    watershed = skimage.segmentation.watershed(
        -pred, labeled, mask=pred>lower_bound)
    for c in range(1, colours + 1):
        response.append({
            "key": '{}.{}.{}'.format(side, key, c),
            "prob": pred[mask].mean(),
            "side": side,
            "type": key,
            "rle": {
                "mask": rle.rle_encode(watershed == c),
                "width": mask.shape[1],
                "height": mask.shape[0],
            }})
    return response


def build_mass_response(channel, threshold=.5, argmax=True):
    mass = addict.Dict()
    mass.response.inhomogen =  np.max([ 
        v['head_predictions'][0].max() for v in channel.values() ])
    mass.response.obscure_margin = np.max([ 
        v['head_predictions'][1].max() for v in channel.values() ])
    mass.response.irregular_shape = np.max([ 
        v['head_predictions'][2].max() for v in channel.values() ])
    mass.response.with_calc = np.max([ 
        v['fpn_predictions'][0].max() for v in channel.values() ])
    mass.response.no = all( v < threshold for v in mass.response.values() )
    mass.response.homogen = 1. - mass.response.inhomogen
    mass.response.circ_margin = 1. - mass.response.obscure_margin
    mass.response.regular_shape = 1. - mass.response.irregular_shape
    mass.response.without_calc = 1. - mass.response.with_calc
    mass.default = "no"
    mass.threshold = threshold
    mass.argmax = argmax
    return mass


def build_density_response(channel, threshold=.5, argmax=True):
    density = addict.Dict()
    density.response.ab = np.median([ 
        v['density'][:2].sum() for v in channel.values() ])
    density.response.bc = 1. - density.response.ab
    density.threshold = threshold
    density.argmax = True
    return density


def build_cancer_prob_response(channel, manager):
    cancer_prob = addict.Dict()
    cancer_prob.response.value = get_cancer_dtr_response(channel, manager)
    cancer_prob.key_value = True
    return cancer_prob


def build_birads_response(channel, manager):
    cancer_prob = addict.Dict()
    value = get_cancer_dtr_response(channel, manager)
    centroids = np.array(config.MAMMOGRAPHY_PARAMS.BIRADS_CENTROIDS)

    tmp = 'birads_{}'
    birads = {}
    values = np.exp((centroids - value)**2)
    for i, v in enumerate(values):
        birads[tmp.format(i + 1)] = v
        birads[tmp.format(0)] = 0

    cancer_prob.response = birads
    cancer_prob.default = 'birads_1',
    cancer_prob.argmax = True,
    cancer_prob.threshold = .5,
    return cancer_prob


def build_distortions_response(channel, threshold=.5, argmax=True):
    distortions = addict.Dict()
    distortions.response.yes = np.max([ 
        v['head_predictions'][4].max() for v in channel.values() ])
    distortions.response.no = 1. - distortions.response.yes
    distortions.threshold = threshold
    distortions.default = "no"
    distortions.argmax = argmax
    return distortions


def build_calcifications_response(channel, threshold=.5, argmax=True):
    calcifications = addict.Dict()
    calcifications.response.no = 1. - np.max([ 
        v['fpn_predictions'][0].max() for v in channel.values() ])
    calcifications.response.malignant = np.max([ 
        v['head_predictions'][3].max() for v in channel.values() ])
    calcifications.response.benign = 1. - calcifications.response.malignant
    calcifications.threshold = threshold
    calcifications.default = "no"
    calcifications.argmax = argmax
    return calcifications


def build_paths_response(channel, channel_id):
    sides = addict.Dict()
    for side, el in channel.items():
        sides[side] = addict.Dict()
        root = config.PATHS.OUTPUT/channel_id/side
        os.makedirs(root, exist_ok=True)
        for i, fpn in enumerate(el['fpn_predictions']):
            path = str(root/'fpn_{}.png'.format(config.MAMMOGRAPHY_PARAMS.NAMES['fpn'][i]))
            sides[side][config.MAMMOGRAPHY_PARAMS.NAMES['fpn'][i]] = path
            cv2.imwrite(path, (fpn * 255).astype(np.uint8))

        for i, head in enumerate(el['head_predictions']):
            path = str(root/'head_{}.png'.format(config.MAMMOGRAPHY_PARAMS.NAMES['head'][i]))
            sides[side][config.MAMMOGRAPHY_PARAMS.NAMES['head'][i]] = path
            cv2.imwrite(path, (head * 255).astype(np.uint8))
    return sides

def build_paths_response_tmp(channel, channel_id):
    sides = addict.Dict()
    for side, el in channel.items():
        root = config.PATHS.OUTPUT/channel_id/side
        os.makedirs(root, exist_ok=True)
        for i, fpn in enumerate(el['fpn_predictions']):
            path = str(root/'fpn_{}.png'.format(config.MAMMOGRAPHY_PARAMS.NAMES['fpn'][i]))
            sides['{}_{}'.format(side, config.MAMMOGRAPHY_PARAMS.NAMES['fpn'][i])] = path
            cv2.imwrite(path, (fpn * 255).astype(np.uint8))

        for i, head in enumerate(el['head_predictions']):
            path = str(root/'head_{}.png'.format(config.MAMMOGRAPHY_PARAMS.NAMES['head'][i]))
            sides['{}_{}'.format(side, config.MAMMOGRAPHY_PARAMS.NAMES['head'][i])] = path
            cv2.imwrite(path, (head * 255).astype(np.uint8))
    return sides


def build_calcifications_findings_response(channel, thresholds):
    findings = list()
    for side, v in channel.items():
        name = config.MAMMOGRAPHY_PARAMS.REVERSE_NAMES['head']['calcification_malignant']
        malignancy = v['head_predictions'][name]
        v['centroids']['malignant'] = 0.
        for i, row in v['centroids'].iterrows():
            v['centroids'].loc[i, 'malignant'] = malignancy[
                int(np.clip(row.unscaled_centroid_y / 16, 0, malignancy.shape[0] - 1)), 
                int(np.clip(row.unscaled_centroid_x / 16, 0, malignancy.shape[1] - 1))
            ]
        for i, row in v['centroids'].iterrows():
            if row.probability < thresholds["calcification"]:
                continue
            tmp = {
                "key": i,
                "scores": {},
                "side": side,
                "type": 'calcification',
                "calcification": row.probability,
                "geometry": {
                    "points": [{
                        "x": int(row.centroid_x),
                        "y": int(row.centroid_y) }]
                }}
            if row.malignant > thresholds['calcification_malignant']:
                tmp.update({"scores": { "malignant": row.malignant }})
            findings.append(tmp)
    return findings

def get_cancer_dtr_response(channel, manager):
    masks = {
        'fpn': {},
        'head': {},
    }

    for side, view in channel.items():
        preds = view['fpn_predictions'].reshape(view['fpn_predictions'].shape[0], -1).max(1)
        for i, el in enumerate(preds):
            masks['fpn'][rt.MASK_NAMES['fpn'][i]] = [el]
        preds = view['head_predictions'].reshape(view['head_predictions'].shape[0], -1).max(1)
        for i, key in rt.MASK_NAMES['head'].items():
            masks['head'][key] = [preds[i]]

    return manager.DecisionTreeRegressor.learner(masks)


def get_rle_response_old(channel):
    findings = list()
    for side, el in channel.items():
        for ptype in ['head', 'fpn']:
            for i, pred in enumerate(el['{}_predictions'.format(ptype)]):
                pred = cv2.resize(
                    (255. * pred).astype(np.uint8),
                    channel[side]['original'].shape[:2][::-1]
                )
                findings.extend(build_rle_findings(
                    pred.astype(np.float) / 255., 
                    config.THRESHOLDS[ptype][i],
                    config.THRESHOLDS_LOWER_BOUND[ptype][i],
                    key=config.MAMMOGRAPHY_PARAMS.NAMES[ptype][i],
                    side=side
                ))
    return findings


def get_rle_response(channel):
    findings = list()
    for side, el in channel.items():
        for ptype in ['head', 'fpn']:
            for i, pred in enumerate(el['{}_predictions'.format(ptype)]):
                findings.extend(build_rle_findings(
                    pred.astype(np.float), 
                    config.THRESHOLDS[ptype][i],
                    config.THRESHOLDS_LOWER_BOUND[ptype][i],
                    key=config.MAMMOGRAPHY_PARAMS.NAMES[ptype][i],
                    side=side
                ))
    return findings


def inter_dice(a, b):
    a = a.astype(np.bool)
    b = b.astype(np.bool)
    return ((a & b).sum() ) / (min(a.sum(), b.sum()) + 1e-5)


def build_findings_rle_response(channel, thresholds):
    findings = list()
    for side, el in channel.items():
        calcification = torch.tensor(el['fpn_predictions'][1:])
        calcification = F.max_pool2d(
            calcification, kernel_size=16, stride=16).data.numpy()[0]

        for mode, mode_keys in config.MAMMOGRAPHY_PARAMS.MODES.items():
            response_mask = np.zeros_like(el['head_predictions'][0])
            response = defaultdict(dict)

            for i, pred in enumerate(el['head_predictions']):
                key = config.MAMMOGRAPHY_PARAMS.NAMES['head'][i]
                if key not in mode_keys:
                    continue

                mask = pred > thresholds[key]
                labeled, colours = scipy.ndimage.label(mask)
                watershed = skimage.segmentation.watershed(
                    -pred, labeled, 
                    mask=pred>config.THRESHOLDS_LOWER_BOUND['head'][i])

                for c in range(1, colours + 1):
                    roin = watershed == c
                    colour = response_mask.max() + 1
                    response[colour].update({ key:  pred[roin].max() })

                    intersected = response_mask[roin]
                    colourso = np.unique(intersected[intersected != 0])

                    mathced = [ co for co, v in { 
                        co: inter_dice(response_mask == co, roin)
                        for co in colourso
                    }.items() if v > .2 ]

                    for co in mathced:
                        roin |= response_mask == co
                        response[colour].update({
                            k: v if k not in response[colour] else max(v, response[colour][k])
                            for k, v in response[co].items()
                        })
                        response.pop(co)

                    response_mask[roin] = colour

            for idx, probs in response.items():
                roi = response_mask == idx
                c_score = calcification[roi].max()
                tmp = {
                    "key": '{}.{}'.format(side, idx),
                    "scores": probs,
                    "side": side,
                    "type": mode,
                    "rle": {
                        "mask": rle.rle_encode(roi),
                        "width": roi.shape[1],
                        "height": roi.shape[0],
                    }
                }
                if c_score > thresholds['calcification']:
                    tmp.update({ "calcification": c_score })
                findings.append(tmp)

    return findings


def build_response(channel, channel_id, manager, thresholds):
    response = addict.Dict()
    response.prediction = addict.Dict()
    response.prediction.density.update(build_density_response(channel))
    response.prediction.distortions.update(build_distortions_response(channel))
    response.prediction.mass.update(build_mass_response(channel))
    response.prediction.calcifications.update(build_calcifications_response(channel))
    response.prediction.cancer_prob.update(build_cancer_prob_response(channel, manager))
    response.prediction.birads.update(build_birads_response(channel, manager))
    response.paths = build_paths_response_tmp(channel, channel_id)
    response.findings = build_calcifications_findings_response(channel, thresholds)
    response.findings.extend(build_findings_rle_response(channel, thresholds))

#     Yura has asked to exclude them
#     response.prediction.foreign_bodies = {
#             "response":
#             {
#                 "no": 0.6030043403,
#                 "skin_mark": 0.033554545,
#                 "breast_implant": 0.0083544105,
#                 "tissue_mark": 0.0083544105
#             },
#             "default": "no",
#             "threshold": 0.5,
#             "argmax": True
#         }

    response.prediction.asymmetry = {
            "response":
            {
                "no": 0.6030043403,
                "local": 0.033554545,
                "total": 0.0083544105,
                "local_calc": 0.0083544105,
                "dynamic": 0.0083544105
            },
            "default": "no",
            "threshold": 0.5,
            "argmax": True
        }
    response.prediction.lymph_node = {
            "response":
            {
                "yes": 0.008354410529136658,
                "no": 0.9916439652442932
            },
            "threshold": 0.5,
            "default": "no",
            "argmax": True
        }
    return response
