from urllib.request import urlopen
import pydicom as dicom
import numpy as np
from numpy import *
import SimpleITK as sitk
import cv2
import os
import re


url_pattern = r'(([\w-]+://?|www[.])[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/)))'

def read_slice_hu(path):
    dcm = dicom.read_file(path)
    image = dcm.pixel_array.astype(np.int16)
    image[image == image.min()] = 0

    # Convert to Hounsfield units (HU)
    if dcm.RescaleSlope != 1:
        image = slope * image.astype(float64)
        image = image.astype(int16)

    image += np.int16(dcm.RescaleIntercept)
    return image


def read_ct_scan(path, verbose=False):
    # type: (object) -> object
    # Read the slices from the dicom file
    slices = list()
    names = list()
    if isinstance(path, list):
        for p in path:
            name, image = read_ct_scan(p)
            slices.append(image)
            names.append(name)
        return names, image

    if re.match(url_pattern, path):
        req = urlopen(path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1) # 'Load it as it is'
        return os.path.basename(path), image

    if os.path.isfile(path):
        try:
            return sitk.ReadImage(path)
        except:
            if verbose:
                print('Neither a DICOM nor a MHD file: %s' % os.path.basename(path))

    if os.path.isdir(path):
        files = os.listdir(path)
        for filename in files:
            try:
                slices.append((
                    filename, dicom.read_file(os.path.join(path, filename))
                ))
            except dicom.filereader.InvalidDicomError:
                if verbose:
                    print('Neither a DICOM nor a MHD file: %s' % filename)

        slices.sort(key=lambda x: int(x[1].InstanceNumber))
        names = [ s[0] for s in slices ]
        slices = [ s[1] for s in slices ]

        try:
            slice_thickness = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except AttributeError:
            slice_thickness = abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        return names, slices


def extract_array(ct_scan):
        heights = asarray([int(ct_slice.SliceLocation)for ct_slice in ct_scan])
        ct_scan = stack([ct_slice.pixel_array for ct_slice in ct_scan])
        ct_scan[ct_scan == ct_scan.min()] = 0
        return ct_scan, heights


def get_pixels_hu(slices):
    try:
        image = stack([s.pixel_array for s in slices])
    except AttributeError:
        return sitk.GetArrayFromImage(slices)
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == image.min()] = 0


    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(float64)
            image[slice_number] = image[slice_number].astype(int16)

        image[slice_number] += int16(intercept)

    return array(image, dtype=int16)
