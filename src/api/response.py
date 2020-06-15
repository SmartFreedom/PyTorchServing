import os
import cv2
import addict
import numpy as np

from src.configs import config


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


def build_findings_response(channel, threshold=.2):
    findings = list()
    for side, v in channel.items():
        for i, row in v['centroids'].iterrows():
            if threshold is not None and row.probability < threshold:
                continue
            findings.append({
                "key": i,
                "prob": row.probability,
                "image": side,
                "geometry": {
                    "points": [{
                        "x": int(row.centroid_x),
                        "y": int(row.centroid_y) }]
                }})
    return findings


def build_response(channel, channel_id):
    response = addict.Dict()
    response.prediction = addict.Dict()
    response.prediction.density.update(build_density_response(channel))
    response.prediction.distortions.update(build_distortions_response(channel))
    response.prediction.mass.update(build_mass_response(channel))
    response.prediction.calcifications.update(build_calcifications_response(channel))
    response.paths = build_paths_response(channel, channel_id)
    response.findings = build_findings_response(channel)
    return response
