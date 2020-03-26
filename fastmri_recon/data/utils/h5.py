"""Utilities to get the data from the h5 files"""
import glob

import h5py


def _from_file_to_stuff(filename, vals=None, attrs=None):
    stuff = []
    if vals is None:
        vals = []
    if attrs is None:
        attrs = []
    with  h5py.File(filename, 'r') as h5_obj:
        for val in vals:
            stuff.append(h5_obj[val][()])
        for attr in attrs:
            stuff.append(h5_obj.attrs[attr])
    if len(stuff) == 1:
        stuff = stuff[0]
    return stuff

def from_test_file_to_mask_and_kspace(filename):
    """Get the mask and kspaces from an h5 file with 'mask' and 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['mask', 'kspace'])


def from_train_file_to_image_and_kspace(filename):
    """Get the images and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['reconstruction_esc', 'kspace'])

def from_train_file_to_image_and_kspace_and_contrast(filename):
    """Get the images and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['reconstruction_esc', 'kspace'], attrs=['acquisition'])

def from_multicoil_train_file_to_image_and_kspace(filename):
    """Get the images and kspaces from an h5 file with 'reconstruction_rss'
    and 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['reconstruction_rss', 'kspace'])

def from_multicoil_train_file_to_image_and_kspace_and_contrast(filename):
    """Get the images and kspaces from an h5 file with 'reconstruction_rss'
    and 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['reconstruction_rss', 'kspace'], attrs=['acquisition'])

def from_test_file_to_mask_and_kspace_and_contrast(filename):
    """Get the mask and kspaces from an h5 file with 'mask'
    and 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['mask', 'kspace'], attrs=['acquisition'])

def from_test_file_to_mask_and_contrast(filename):
    """Get the mask and kspaces from an h5 file with 'mask'
    and 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['mask'], attrs=['acquisition'])


def from_file_to_kspace(filename):
    """Get the kspaces from an h5 file with 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['kspace'])

def from_file_to_contrast(filename):
    """Get the contrast from an h5 file.
    """
    return _from_file_to_stuff(filename, attrs=['acquisition'])

def list_files_w_contrast_and_af(path, AF=4, contrast=None):
    filenames = glob.glob(path + '*.h5')
    filenames.sort()
    for filename in filenames:
        mask, contrast_f = from_test_file_to_mask_and_contrast(filename)
        if contrast and contrast != contrast_f:
            continue
        mask_af = len(mask) / sum(mask)
        if AF == 4 and mask_af < 5.5 or AF == 8 and mask_af > 5.5:
            yield filename
