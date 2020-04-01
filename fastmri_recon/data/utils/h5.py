"""Utilities to get the data from the h5 files"""
import glob
import random

import h5py


def _from_file_to_stuff(filename, vals=None, attrs=None, selection=None):
    stuff = []
    if vals is None:
        vals = []
    if attrs is None:
        attrs = []
    if selection is None:
        selection = {}
    with h5py.File(filename, 'r') as h5_obj:
        for val in vals:
            h5_dataset = h5_obj[val]
            if val in selection:
                data_shape = h5_dataset.shape
                dimensions_selection_list = selection[val]
                selected_slices_list = []
                for i_dimension, dimension_selection in enumerate(dimensions_selection_list):
                    data_dimension = data_shape[i_dimension]
                    rand = dimension_selection.get('rand', False)
                    inner_slices = dimension_selection.get('inner_slices', None)
                    if inner_slices is not None:
                        slice_start = data_dimension // 2 - inner_slices // 2
                        slice_end = slice_start + inner_slices - 1
                    else:
                        slice_start = 0
                        slice_end = data_dimension
                    if rand:
                        i_slice = random.randint(slice_start, slice_end)
                        selected_slices = slice(i_slice, i_slice + 1)
                    else:
                        selected_slices = slice(slice_start, slice_end)
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

def from_test_file_to_mask_and_kspace(filename, selection=None):
    """Get the mask and kspaces from an h5 file with 'mask' and 'kspace' keys.
    """
    if selection is not None:
        selection = {'kspace': selection}
    return _from_file_to_stuff(filename, vals=['mask', 'kspace'], selection=selection)


def from_train_file_to_image_and_kspace(filename, selection=None):
    """Get the images and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    if selection is not None:
        selection = {'kspace': selection, 'reconstruction_esc': selection}
    return _from_file_to_stuff(filename, vals=['reconstruction_esc', 'kspace'], selection=selection)

def from_train_file_to_image_and_kspace_and_contrast(filename, selection=None):
    """Get the images and kspaces from an h5 file with 'reconstruction_esc'
    and 'kspace' keys.
    """
    if selection is not None:
        selection = {'kspace': selection, 'reconstruction_esc': selection}
    return _from_file_to_stuff(filename, vals=['reconstruction_esc', 'kspace'], attrs=['acquisition'], selection=selection)

def from_multicoil_train_file_to_image_and_kspace(filename, selection=None):
    """Get the images and kspaces from an h5 file with 'reconstruction_rss'
    and 'kspace' keys.
    """
    if selection is not None:
        selection = {'kspace': selection, 'reconstruction_rss': selection}
    return _from_file_to_stuff(filename, vals=['reconstruction_rss', 'kspace'], selection=selection)

def from_multicoil_train_file_to_image_and_kspace_and_contrast(filename, selection=None):
    """Get the images and kspaces from an h5 file with 'reconstruction_rss'
    and 'kspace' keys.
    """
    if selection is not None:
        selection = {'kspace': selection, 'reconstruction_rss': selection}
    return _from_file_to_stuff(filename, vals=['reconstruction_rss', 'kspace'], attrs=['acquisition'], selection=selection)

def from_test_file_to_mask_and_kspace_and_contrast(filename, selection=None):
    """Get the mask and kspaces from an h5 file with 'mask'
    and 'kspace' keys.
    """
    if selection is not None:
        selection = {'kspace': selection}
    return _from_file_to_stuff(filename, vals=['mask', 'kspace'], attrs=['acquisition'], selection=selection)

def from_test_file_to_mask_and_contrast(filename):
    """Get the mask and kspaces from an h5 file with 'mask'
    and 'kspace' keys.
    """
    return _from_file_to_stuff(filename, vals=['mask'], attrs=['acquisition'])


def from_file_to_kspace(filename, selection=None):
    """Get the kspaces from an h5 file with 'kspace' keys.
    """
    if selection is not None:
        selection = {'kspace': selection}
    return _from_file_to_stuff(filename, vals=['kspace'], selection=selection)

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
