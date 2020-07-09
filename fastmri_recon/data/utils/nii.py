import nibabel as nib
import numpy as np


def from_file_to_volume(filename):
    volume = nib.load(filename)
    volume = volume.get_fdata()
    volume = volume[..., None]
    volume = volume.astype(np.complex64)
    return volume
