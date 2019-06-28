class FFT2:
    def __init__(self, mask):
        self.mask = mask

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
        fft_coeffs = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img), norm='ortho'))
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
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(masked_fft_coeffs), norm='ortho'))
