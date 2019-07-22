import random

from keras import backend as K
from keras.callbacks import Callback
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

# taken from https://github.com/facebookresearch/fastMRI/
def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std

# TODO have the same in 3D
def keras_psnr(y_true, y_pred):
    max_pixel = K.max(y_true)
    return tf.image.psnr(y_true, y_pred, max_pixel)

def keras_ssim(y_true, y_pred):
    max_pixel = K.max(y_true)
    return tf.image.ssim(y_true, y_pred, max_pixel)


class VolumeMetrics(Callback):
    def __init__(self, verbose=0):
        super(VolumeMetrics, self).__init__()
        self.verbose = verbose

    def on_train_begin(self):
        self.train_psnrs = list()
        self.train_ssims = list()
        self.val_psnrs = list()
        self.val_ssims = list()

    def on_batch_end(self, batch, logs=None):
        y_pred = self.model.predict(self.model.training_data[0])
        psnr = keras_psnr(self.model.training_data[1], y_pred)
        self.train_psnrs.append(psnr)
        ssim = keras_ssim(self.model.training_data[1], y_pred)
        self.train_ssims.append(ssim)
        if self.verbose > 1:
            print('Training PSNR at batch {batch}: {self.train_psnrs[-1]}')
            print('Training SSIM at batch {batch}: {self.train_ssims[-1]}')

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.model.validation_data[0])
        psnr = keras_psnr(self.model.validation_data[1], y_pred)
        self.val_psnrs.append(psnr)
        ssim = keras_ssim(self.model.validation_data[1], y_pred)
        self.val_ssims.append(ssim)
        if self.verbose > 0:
            print('Validation PSNR at epoch {epoch}: {self.val_psnrs[-1]}')
            print('Validation SSIM at epoch {epoch}: {self.val_ssims[-1]}')
