"""Utilities to get the data from the h5 files"""
import glob
import random

import h5py


def _from_file_to_stuff(filename, vals=None, attrs=None, rand_slices=None):
    stuff = []
    if vals is None:
        vals = []
    if attrs is None:
        attrs = []
    # rand_slices is a dictionary specifying the inner slices you want to
    # consider for random sampling. If None, all slices are considered for
    # random slicing. Slice is here in the Python meaning not MRI, which means
    # that coils are also subject to slicing.
    if rand_slices is None:
        rand_slices = {}
    with h5py.File(filename, 'r') as h5_obj:
        for val in vals:
            h5_dataset = h5_obj[val]
            if val in rand_slices:
                data_shape = h5_dataset.shape
                inner_slices_list = rand_slices[val]
                selected_slices_list = []
                for i_inner_slice, inner_slices in enumerate(inner_slices_list):
                    data_dimension = data_shape[i_inner_slice]
                    if inner_slices is None:
                        # all slices need to be considered for random slicing
                        i_slice = random.randint(0, data_dimension)
                    else:
                        # only inner slices need to be considered for random
                        # slicing
                        slice_start = data_dimension // 2 - inner_slices // 2
                        i_slice = random.randint(slice_start, slice_start + inner_slices - 1)
                    selected_slices = slice(i_slice, i_slice + 1)
                    selected_slices_list.append(selected_slices)
                selected_slices_tuple = tuple(selected_slices_list)
                stuff.append(h5_dataset[selected_slices_tuple])
            else:
                stuff.append(h5_dataset[()])
        for attr in attrs:
            stuff.append(h5_obj.attrs[attr])
    if len(stuff) == 1:
        stuff = stuff[0]
    return stuff

def from_test_file_to_mask_and_kspace(filename, rand_slices=None):
    """Get the mask and kspaces from an h5 file with 'mask' and 'kspace' keys.
    """
    if rand_slices is not None:
        rand_slices = {'kspace': rand_slices}
    return _from_file_to_stuff(filename, vals=['mask', 'kspace'], rand_slices=rand_slices)


def from_train_file_to_image_and_kspace(filename, rand_slices=None):
    """Get the images and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    if rand_slices is not None:
        rand_slices = {'kspace': rand_slices, 'reconstruction_esc': rand_slices}
    return _from_file_to_stuff(filename, vals=['reconstruction_esc', 'kspace'], rand_slices=rand_slices)

def from_train_file_to_image_and_kspace_and_contrast(filename, rand_slices=None):
    """Get the images and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    if rand_slices is not None:
        rand_slices = {'kspace': rand_slices, 'reconstruction_esc': rand_slices}
    return _from_file_to_stuff(filename, vals=['reconstruction_esc', 'kspace'], attrs=['acquisition'], rand_slices=rand_slices)

def from_multicoil_train_file_to_image_and_kspace(filename, rand_slices=None):
    """Get the images and kspaces from an h5 file with 'reconstruction_rss'
    and 'kspace' keys.
    """
    if rand_slices is not None:
        rand_slices = {'kspace': rand_slices, 'reconstruction_rss': rand_slices}
    return _from_file_to_stuff(filename, vals=['reconstruction_rss', 'kspace'], rand_slices=rand_slices)

def from_multicoil_train_file_to_image_and_kspace_and_contrast(filename, rand_slices=None):
    """Get the images and kspaces from an h5 file with 'reconstruction_rss'
    and 'kspace' keys.
    """
    if rand_slices is not None:
        rand_slices = {'kspace': rand_slices, 'reconstruction_rss': rand_slices}
    return _from_file_to_stuff(filename, vals=['reconstruction_rss', 'kspace'], attrs=['acquisition'], rand_slices=rand_slices)

def from_test_file_to_mask_and_kspace_and_contrast(filename, rand_slices=None):
    """Get the mask and kspaces from an h5 file with 'mask'
    and 'kspace' keys.
    """
    if rand_slices is not None:
        rand_slices = {'kspace': rand_slices}
    return _from_file_to_stuff(filename, vals=['mask', 'kspace'], attrs=['acquisition'], rand_slices=rand_slices)

def from_test_file_to_mask_and_contrast(filename):
    """Get the mask and kspaces from an h5 file with 'mask'
    and 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['mask'], attrs=['acquisition'])


def from_file_to_kspace(filename, rand_slices=None):
    """Get the kspaces from an h5 file with 'kspace' keys.
    """
    if rand_slices is not None:
        rand_slices = {'kspace': rand_slices}
    return _from_file_to_stuff(filename, vals=['kspace'], rand_slices=rand_slices)

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
