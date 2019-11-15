import numpy as np
import sklearn.cluster
import skimage.draw
import scipy.ndimage
import cv2
import re

from ..configs import config

pattern = re.compile(r'^[\sa-zA-Z_0]*([0-9]+)\.dcm')


def names_to_indexes(names):
    subs = [re.sub(pattern, r'\1', name) for name in names]
    return list(map(int, subs))


def polygon2mask(shape, vertex_row_coords, vertex_col_coords):
    fill_row_coords, fill_col_coords = skimage.draw.polygon(
        vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def cluster_roi(scan_slice, mask):
    cluster = sklearn.cluster.KMeans(n_clusters=2)
    roi = scan_slice[mask].reshape(-1, 1)
    labels = cluster.fit_predict(roi.copy()).astype(np.bool_)
    nlabels = np.logical_not(labels)

    if roi[labels].mean() < roi[nlabels].mean():
        labels = nlabels

    mask_ = mask.copy()
    mask_[mask] = labels
    return mask_


def get_polygon(scan, annot, idxs):
    annot = annot.sort_values('slice')
    coords = annot.coords.values[1 if len(annot) > 1 else 0]
    slice_ = annot.slice.values[1 if len(annot) > 1 else 0]
    predicat = True
    for i, c in annot.iterrows():
        if len(c.coords[::2]) > 1:
            coords = c.coords.copy()
            slice_ = c.slice
            predicat = False

    idx = idxs.index(slice_)
    mask = polygon2mask(scan.shape[1:],
                        coords[1::2] + coords[1:2],
                        coords[::2] + coords[:1]
                        )
    if predicat:
        mask[coords[1], coords[0]] = 1
        return scipy.ndimage.binary_dilation(mask, iterations=10), idx
    return mask.astype(np.bool_), idx


def save_mask(mask, names, case):
    idxs = np.unique(np.where(mask > 1)[0])
    names = [names[i][:-4] + ".png" for i in idxs]
    for idx, name in zip(idxs, names):
        try:
            os.mkdir(config.PATHS.MASKS / case)
        except:
            pass
        cv2.imwrite(str(config.PATHS.MASKS / case / name), mask[idx].astype(np.uint8))
