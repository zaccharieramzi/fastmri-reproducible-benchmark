"""Fourier utilities"""
import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.signal.fft_ops import ifft2d, ifftshift, fftshift


class FFT2:
    """This class defines the masked fourier transform operator in 2D, where
    the mask is defined on shifted fourier coefficients.
    """
    def __init__(self, mask):
        self.mask = mask
        self.shape = mask.shape

    def op(self, img):
        """ This method calculates the masked Fourier transform of a 2-D image.

        Parameters
        ----------
        img: np.ndarray
            input 2D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        fft_coeffs = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))
        return self.mask * fft_coeffs

    def adj_op(self, x):
        """ This method calculates inverse masked Fourier transform of a 2-D
        image.

        Parameters
        ----------
        x: np.ndarray
            masked Fourier transform data.

        Returns
        -------
        img: np.ndarray
            inverse 2D discrete Fourier transform of the input coefficients.
        """
        masked_fft_coeffs = self.mask * x
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(masked_fft_coeffs, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def fft(image):
    """Perform the fft of an image"""
    fourier_op = FFT2(np.ones_like(image))
    kspace = fourier_op.op(image)
    return kspace

def ifft(kspace):
    """Perform the ifft of an image"""
    fourier_op = FFT2(np.ones_like(kspace))
    image = fourier_op.adj_op(kspace)
    return image

def tf_ortho_ifft2d(kspace, enable_multiprocessing=True):
    axes = [len(kspace.shape) - 2, len(kspace.shape) - 1]
    scaling_norm = tf.cast(tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32')), kspace.dtype)
    if len(kspace.shape) == 4:
        # multicoil case
        ncoils = tf.shape(kspace)[1]
    n_slices = tf.shape(kspace)[0]
    k_shape_x = tf.shape(kspace)[-2]
    k_shape_y = tf.shape(kspace)[-1]
    shifted_kspace = ifftshift(kspace, axes=axes)
    if enable_multiprocessing:
        batched_shifted_kspace = tf.reshape(shifted_kspace, (-1, k_shape_x, k_shape_y))
        batched_shifted_image = tf.map_fn(
            ifft2d,
            batched_shifted_kspace,
            parallel_iterations=multiprocessing.cpu_count(),
        )
        if len(kspace.shape) == 4:
            # multicoil case
            image_shape = [n_slices, ncoils, k_shape_x, k_shape_y]
        elif len(kspace.shape) == 3:
            image_shape = [n_slices, k_shape_x, k_shape_y]
        else:
            image_shape = [k_shape_x, k_shape_y]
        shifted_image = tf.reshape(batched_shifted_image, image_shape)
    else:
        shifted_image = ifft2d(shifted_kspace)
    image = fftshift(shifted_image, axes=axes)
    return scaling_norm * image
