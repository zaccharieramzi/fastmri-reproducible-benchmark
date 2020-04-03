import numpy as np
import tensorflow as tf

from fastmri_recon.data.utils.fourier import FFT2, ifft
from fastmri_recon.data.utils.masking.gen_mask import gen_mask
from fastmri_recon.models.utils.fourier import FFT, IFFT


class TestFFTLayers(tf.test.TestCase):
    def setUp(self):
        kspace_shape = (640, 372)
        n_coils = 15
        self.kspace = np.random.normal(size=kspace_shape) + 1j * np.random.normal(size=kspace_shape)
        self.image = ifft(self.kspace)
        self.smaps = np.random.normal(size=(n_coils,) + kspace_shape) + 1j * np.random.normal(size=(n_coils,) +kspace_shape)
        mask = gen_mask(self.kspace)
        self.mask = np.repeat(mask.astype(np.float), kspace_shape[0], axis=0)
        fourier_op = FFT2(self.mask)
        self.masked_kspace = fourier_op.op(self.image)
        self.masked_kspace_multi_coil = fourier_op.op(self.image[None, ...] * self.smaps)
        self.aliased_image = fourier_op.adj_op(self.kspace)
        self.aliased_image_sense = np.sum(fourier_op.adj_op(self.masked_kspace_multi_coil) * np.conj(self.smaps), axis=0)

    def test_fft(self):
        image_tf = tf.convert_to_tensor(self.image)[None, ..., None]
        mask_tf = tf.convert_to_tensor(self.mask)[None, ...]
        fourier_layer = FFT(masked=True)
        masked_kspace = fourier_layer([image_tf, mask_tf])
        self.assertAllClose(self.masked_kspace, masked_kspace[0, ..., 0])

    def test_fft_multi_coil(self):
        image_tf = tf.convert_to_tensor(self.image)[None, ..., None]
        mask_tf = tf.convert_to_tensor(self.mask)[None, ...]
        smaps_tf = tf.convert_to_tensor(self.smaps)[None, ...]
        fourier_layer = FFT(masked=True, multicoil=True)
        masked_kspace_multicoil = fourier_layer([image_tf, mask_tf, smaps_tf])
        self.assertAllClose(self.masked_kspace_multi_coil, masked_kspace_multicoil[0, ..., 0])

    def test_ifft(self):
        kspace_tf = tf.convert_to_tensor(self.kspace)[None, ..., None]
        mask_tf = tf.convert_to_tensor(self.mask)[None, ...]
        inverse_fourier_layer = IFFT(masked=True)
        aliased_image = inverse_fourier_layer([kspace_tf, mask_tf])
        self.assertAllClose(self.aliased_image, aliased_image[0, ..., 0])

    def test_adj_fft_multi_coil(self):
        masked_kspace_multi_coil_tf = tf.convert_to_tensor(self.masked_kspace_multi_coil)[None, ..., None]
        mask_tf = tf.convert_to_tensor(self.mask)[None, ...]
        smaps_tf = tf.convert_to_tensor(self.smaps)[None, ...]
        inverse_fourier_layer = IFFT(masked=True, multicoil=True)
        aliased_image_sense = inverse_fourier_layer([masked_kspace_multi_coil_tf, mask_tf, smaps_tf])
        self.assertAllClose(self.aliased_image_sense, aliased_image_sense[0, ..., 0])
