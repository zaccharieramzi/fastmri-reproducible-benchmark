from pathlib import Path

import numpy as np
import tensorflow as tf

from fastmri_recon.data.utils.h5 import from_test_file_to_mask_and_contrast, from_recon_file_to_reconstruction


def compute_af(mask):
    return len(mask) / np.sum(mask)

def tf_filename_to_recon(filename):
    def _from_recon_file_to_recon_tensor_to_tensor(filename):
        filename_str = filename.numpy()
        recon = from_recon_file_to_reconstruction(filename_str)
        recon = recon[..., None]
        return tf.convert_to_tensor(recon)
    [recon] = tf.py_function(
        _from_recon_file_to_recon_tensor_to_tensor,
        [filename],
        [tf.float32],
    )
    recon.set_shape((None, None, None, None))
    return recon

class PostprocH5DatasetBuilder:
    def __init__(self, orig_path, recon_path, af=4, contrast=None, prefetch=True):
        self.orig_path = Path(orig_path)
        self.recon_path = Path(recon_path)
        self.af = af
        self.contrast = contrast
        self.prefetch = prefetch
        self.num_parallel_calls = 2
        self.orig_filenames = sorted(list(self.orig_path.glob('*.h5')))
        self.recon_filenames = sorted(list(self.recon_path.glob('*.h5')))
        self.filter_filenames()
        self.files_ds = tf.data.Dataset.from_tensor_slices(
            [str(f) for f in self.recon_filenames],
        )
        self.raw_ds = self.files_ds.map(
            tf_filename_to_recon,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=True,
        )
        if self.prefetch:
            self.raw_ds = self.raw_ds.prefetch(tf.data.experimental.AUTOTUNE)

    def validate_af(self, mask):
        if self.af == 4:
            return compute_af(mask) < 5.5
        else:
            return compute_af(mask) >= 5.5

    def validate_contrast(self, contrast):
        return contrast == self.contrast or self.contrast is None

    def filter_filenames(self):
        masks_and_contrasts = [
            from_test_file_to_mask_and_contrast(filename)
            for filename in self.orig_filenames
        ]
        flags = [
            self.validate_af(mask) and self.validate_contrast(contrast)
            for mask, contrast in masks_and_contrasts
        ]
        self.orig_filenames = [f for i, f in enumerate(self.orig_filenames) if flags[i]]
        self.recon_filenames = [f for i, f in enumerate(self.recon_filenames) if flags[i]]
