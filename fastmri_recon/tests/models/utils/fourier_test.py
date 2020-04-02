import numpy as np
import tensorflow as tf

from fastmri_recon.data.utils.fourier import FFT2, ifft
from fastmri_recon.data.utils.masking.gen_mask import gen_mask
from fastmri_recon.models.utils.fourier import FFT, IFFT


class TestFFTLayers(tf.test.TestCase):
    def setUp(self):
        kspace_shape = (640, 372)
        self.kspace = np.random.normal(size=kspace_shape) + 1j * np.random.normal(size=kspace_shape)
        self.image = ifft(self.kspace)
        mask = gen_mask(self.kspace)
        self.mask = np.repeat(mask.astype(np.float), kspace_shape[0], axis=0)
        fourier_op = FFT2(self.mask)
        self.masked_kspace = fourier_op.op(self.image)
        self.aliased_image = fourier_op.adj_op(self.kspace)

    def test_fft(self):
        image_tf = tf.convert_to_tensor(self.image)[None, ..., None]
        mask_tf = tf.convert_to_tensor(self.mask)[None, ...]
        fourier_layer = FFT(masked=True)
        masked_kspace = fourier_layer([image_tf, mask_tf])
        self.assertAllClose(self.masked_kspace, masked_kspace[0, ..., 0])

    def test_ifft(self):
        kspace_tf = tf.convert_to_tensor(self.kspace)[None, ..., None]
        mask_tf = tf.convert_to_tensor(self.mask)[None, ...]
        inverse_fourier_layer = IFFT(masked=True)
        aliased_image = inverse_fourier_layer([kspace_tf, mask_tf])
        self.assertAllClose(self.aliased_image, aliased_image[0, ..., 0])
