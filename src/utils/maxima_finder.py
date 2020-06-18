import scipy.ndimage
import gc

import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt


def find_maxima(
    data, seriesuid, neighborhood=25, 
    threshold=10/255., scale_factor=1., verbose=False):

    data_max = filters.maximum_filter(data, neighborhood)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    if isinstance(scale_factor, (int, float)):
        scale_factor = [scale_factor] * 2

    labeled, colours = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    centroids = list()
    for colour in range(1, colours + 1):
        roi = labeled == colour
        centroid = np.mean(np.array(np.where(roi)), 1)
        value = data[roi].max()
        centroids.append({ 
            'centroid_x': centroid[1] * scale_factor[1], 
            'centroid_y': centroid[0] * scale_factor[0], 
            'probability': value,
            'seriesuid': seriesuid
        })

    centroids = sorted(centroids, key=lambda x: x['probability'], reverse=True)
    if verbose:
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        ax[0].imshow(data)
        ax[1].imshow(data)
        ax[1].autoscale(False)
        ax[1].plot(
            [c['centroid_x'] / scale_factor[1] for c in centroids[:30]], 
            [c['centroid_y'] / scale_factor[0] for c in centroids[:30]], 'ro')
        plt.show()
    return centroids