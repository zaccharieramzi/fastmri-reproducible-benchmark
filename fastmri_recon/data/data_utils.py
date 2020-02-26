"""Utilities to get the data from the h5 files"""
import glob

import h5py


# TODO: refactor all of this !
def from_test_file_to_mask_and_kspace(filename):
    """Get the mask and kspaces from an h5 file with 'mask' and 'kspace' keys.
    """
    with  h5py.File(filename, 'r') as h5_obj:
        masks = h5_obj['mask'][()]
        kspaces = h5_obj['kspace'][()]
    return masks, kspaces


def from_train_file_to_image_and_kspace(filename):
    """Get the imagess and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    with h5py.File(filename, 'r') as h5_obj:
        images = h5_obj['reconstruction_esc'][()]
        kspaces = h5_obj['kspace'][()]
    return images, kspaces

def from_train_file_to_image_and_kspace_and_contrast(filename):
    """Get the imagess and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    with h5py.File(filename, 'r') as h5_obj:
        images = h5_obj['reconstruction_esc'][()]
        kspaces = h5_obj['kspace'][()]
        contrast = h5_obj.attrs['acquisition']
    return images, kspaces, contrast

def from_test_file_to_mask_and_kspace_and_contrast(filename):
    """Get the imagess and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    with h5py.File(filename, 'r') as h5_obj:
        masks = h5_obj['mask'][()]
        kspaces = h5_obj['kspace'][()]
        contrast = h5_obj.attrs['acquisition']
    return masks, kspaces, contrast

def from_test_file_to_mask_and_contrast(filename):
    """Get the imagess and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    with h5py.File(filename, 'r') as h5_obj:
        masks = h5_obj['mask'][()]
        contrast = h5_obj.attrs['acquisition']
    return masks, contrast

def from_file_to_kspace(filename):
    """Get the kspaces from an h5 file with 'kspace' keys.
    """
    with h5py.File(filename, 'r') as h5_obj:
        kspaces = h5_obj['kspace'][()]
        return kspaces

def from_file_to_contrast(filename):
    """Get the contrast from an h5 file.
    """
    with h5py.File(filename, 'r') as h5_obj:
        contrast = h5_obj.attrs['acquisition']
        return contrast

def list_files_w_contrast_and_af(path, AF=4, contrast=None):
    filenames = glob.glob(path + '*.h5')
    for filename in filenames:
        mask, contrast_f = from_test_file_to_mask_and_contrast(filename)
        if contrast and contrast != contrast_f:
            continue
        mask_af = len(mask) / sum(mask)
        if AF == 4 and mask_af < 5.5 or AF == 8 and mask_af > 5.5:
            yield filename
