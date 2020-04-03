import numpy as np
import tensorflow as tf

from fastmri_recon.data.utils.fourier import ifft
from fastmri_recon.data.utils.multicoil.smap_extract import extract_smaps


class SmapsTest(tf.test.TestCase):
    def setUp(self):
        kspace_shape = (15, 640, 372)
        self.kspace = np.random.normal(size=kspace_shape) + 1j * np.random.normal(size=kspace_shape)
        self.low_freq_percentage = 8
        low_freq_locations = []
        for n in self.kspace.shape[-2:]:
            n_low_freq = int(self.low_freq_percentage * n / 100)
            center_dim = n//2
            low_freq_locations.append(slice(center_dim - n_low_freq//2, center_dim + n_low_freq//2))
        low_freq_mask = np.zeros_like(self.kspace)
        low_freq_mask[:, low_freq_locations[0], low_freq_locations[1]] = 1
        low_freq_kspace = self.kspace * low_freq_mask
        low_freq_ifft = ifft(low_freq_kspace)
        low_freq_rss = np.linalg.norm(low_freq_ifft, axis=0)
        self.coil_smap = low_freq_ifft / low_freq_rss

    def test_extract_smaps(self):
        kspace_tf = tf.convert_to_tensor(self.kspace)[None, ...]
        coil_smap_tf = extract_smaps(kspace_tf, self.low_freq_percentage)
        self.assertAllClose(self.coil_smap, coil_smap_tf[0])
