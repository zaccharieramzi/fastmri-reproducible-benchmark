import random

from keras import backend as K
import numpy as np
import tensorflow as tf


def crop_center(img, cropx, cropy=None):
    # taken from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image/39382475
    if cropy is None:
        cropy = cropx
    y, x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def gen_mask(kspace, accel_factor=8):
    # inspired by https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
    shape = kspace.shape
    num_cols = shape[-1]

    center_fraction = (32 // accel_factor) / 100
    acceleration = accel_factor

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = np.random.uniform(size=num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)
    return mask

# TODO have the same in 3D
def keras_psnr(y_true, y_pred):
    max_pixel = K.max(y_true)
    return tf.image.psnr(y_true, y_pred, max_pixel)

def keras_ssim(y_true, y_pred):
    max_pixel = K.max(y_true)
    return tf.image.ssim(y_true, y_pred, max_pixel)
