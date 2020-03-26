"""Fourier utilities"""
import numpy as np


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
