import nibabel as nib
import numpy as np


def from_file_to_volume(filename, reverse=True):
    # the smallest dimension is the ear-to-ear direction it appears
    # the other dimensions seem to always be in the following order:
    # back-to-front then top-to-bottom
    volume = nib.load(filename)
    volume = volume.get_fdata()
    if reverse and min(volume.shape) != volume.shape[0]:
        volume = np.moveaxis(volume, -1, 0)
    volume = volume.astype(np.complex64)
    return volume
