import numpy as np
import pytest

from fastmri_recon.data.utils.crop import crop_center
from fastmri_recon.data.utils.fourier import ifft
from fastmri_recon.data.utils.h5 import *


def test_all_functions_single_coil(create_full_fastmri_test_tmp_dataset):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    test_file_single_coil = path + 'train_singlecoil_0.h5'
    functions_sliceable = [
        from_train_file_to_image_and_kspace,
        from_train_file_to_image_and_kspace_and_contrast,
        from_file_to_kspace,
    ]

    functions_to_test = functions_sliceable + [from_file_to_contrast]
    for fun in functions_to_test:
        fun(test_file_single_coil)
    for fun in functions_sliceable:
        # this gets the 10-th slice of the volume
        fun(test_file_single_coil, selection=[{'inner_slices': 8, 'rand': True}])
        fun(test_file_single_coil, selection=[{'inner_slices': 8}])

@pytest.mark.skip(reason="No consistency in fake dataset yet")
@pytest.mark.parametrize('selection', [
    [],
    [{'inner_slices': 8, 'rand': True}],
    [{'inner_slices': 8}],
])
def test_from_train_file_to_image_and_kspace(create_full_fastmri_test_tmp_dataset, selection):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    test_file_single_coil = path + 'train_singlecoil_0.h5'
    image, kspace = from_train_file_to_image_and_kspace(test_file_single_coil, selection=selection)
    reconstructed_image = crop_center(np.abs(ifft(kspace)), 320)
    np.testing.assert_allclose(reconstructed_image, image, rtol=1e-4, atol=1e-11)

def test_all_functions_multi_coil(create_full_fastmri_test_tmp_dataset):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_train']
    test_file_multi_coil = path + 'train_multicoil_0.h5'
    functions_sliceable = [
        from_multicoil_train_file_to_image_and_kspace,
        from_multicoil_train_file_to_image_and_kspace_and_contrast,
        from_file_to_kspace,
    ]

    functions_to_test = functions_sliceable + [from_file_to_contrast]
    for fun in functions_to_test:
        fun(test_file_multi_coil)
    for fun in functions_sliceable:
        # this gets the 10-th slice of the volume
        fun(test_file_multi_coil, selection=[{'inner_slices': 8, 'rand': True}, {'rand': True, 'keep_dim': False}])

@pytest.mark.skip(reason="No consistency in fake dataset yet")
@pytest.mark.parametrize('selection', [
    [],
    [{'inner_slices': 8, 'rand': True}],
    [{'inner_slices': 8}],
])
def test_from_multicoil_train_file_to_image_and_kspace(create_full_fastmri_test_tmp_dataset, selection):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_train']
    test_file_multi_coil = path + 'train_multicoil_0.h5'
    image, kspace = from_multicoil_train_file_to_image_and_kspace(test_file_multi_coil, selection=selection)
    reconstructed_image = crop_center(np.linalg.norm(ifft(kspace), axis=1), 320)
    np.testing.assert_allclose(reconstructed_image, image, rtol=1e-4, atol=1e-11)

def test_from_multicoil_train_file_to_image_and_kspace_parallel(create_full_fastmri_test_tmp_dataset):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_train']
    test_file_multi_coil = path + 'train_multicoil_0.h5'
    selection = [{'inner_slices': 8, 'rand': True}, {'rand': True, 'keep_dim': False}]
    _, kspace = from_multicoil_train_file_to_image_and_kspace(test_file_multi_coil, selection=selection)
    assert len(kspace.shape) == 3
