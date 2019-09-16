"""Utilities to get the data from the h5 files"""
from functools import lru_cache

import h5py


def from_test_file_to_mask_and_kspace(filename):
    """Get the mask and kspaces from an h5 file with 'mask' and 'kspace' keys.
    """
    with  h5py.File(filename) as h5_obj:
        masks = h5_obj['mask'][()]
        kspaces = h5_obj['kspace'][()]
    return masks, kspaces


@lru_cache(maxsize=128)
def from_train_file_to_image_and_kspace(filename):
    """Get the imagess and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    with h5py.File(filename) as h5_obj:
        images = h5_obj['reconstruction_esc'][()]
        kspaces = h5_obj['kspace'][()]
    return images, kspaces


def from_file_to_kspace(filename):
    """Get the kspaces from an h5 file with 'kspace' keys.
    """
    with h5py.File(filename) as h5_obj:
        kspaces = h5_obj['kspace'][()]
        return kspaces
