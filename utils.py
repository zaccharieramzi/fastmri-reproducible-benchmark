import random

import numpy as np

def crop_center(img, cropx, cropy=None):
    # taken from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image/39382475
    if cropy is None:
        cropy = cropx
    y, x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def gen_mask(kspace, accel_factor=8):
    n_samples = kspace.shape[-1] // accel_factor
    mask = np.zeros((kspace.shape[-1],)).astype(bool)
    n_center = kspace.shape[-1] * (32 // accel_factor) // 100
    n_non_center = n_samples - n_center
    center_slice = (len(mask)//2 - n_center // 2, len(mask)//2 + n_center // 2)
    mask[slice(*center_slice)] = True
    selected_indices = random.sample(
        [i for i in range(0, kspace.shape[-1]) if i not in range(center_slice[0], center_slice[1])],
        n_non_center,
    )
    mask[selected_indices] = True
    return mask
