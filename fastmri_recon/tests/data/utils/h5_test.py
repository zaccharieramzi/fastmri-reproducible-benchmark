import os

import numpy as np
import pytest

from fastmri_recon.data.utils.crop import crop_center
from fastmri_recon.data.utils.fourier import ifft
from fastmri_recon.data.utils.h5 import *

test_file_single_coil = 'fastmri_recon/tests/fastmri_data/single_coil/file1000002.h5'
test_file_multi_coil = 'fastmri_recon/tests/fastmri_data/multi_coil/file1000001.h5'

@pytest.mark.skipif(not os.path.isfile(test_file_single_coil), reason='test single coil file not present for h5 utils.')
def test_all_functions_single_coil():
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

@pytest.mark.skipif(not os.path.isfile(test_file_single_coil), reason='test single coil file not present for h5 utils.')
@pytest.mark.parametrize('selection', [
    [],
    [{'inner_slices': 8, 'rand': True}],
    [{'inner_slices': 8}],
])
def test_from_train_file_to_image_and_kspace(selection):
    image, kspace = from_train_file_to_image_and_kspace(test_file_single_coil, selection=selection)
    reconstructed_image = crop_center(np.abs(ifft(kspace)), 320)
    np.testing.assert_allclose(reconstructed_image, image, rtol=1e-4, atol=1e-11)

@pytest.mark.skipif(not os.path.isfile(test_file_multi_coil), reason='test multi coil file not present for h5 utils.')
def test_all_functions_multi_coil():
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

@pytest.mark.skipif(not os.path.isfile(test_file_multi_coil), reason='test multi coil file not present for h5 utils.')
@pytest.mark.parametrize('selection', [
    [],
    [{'inner_slices': 8, 'rand': True}],
    [{'inner_slices': 8}],
])
def test_from_multicoil_train_file_to_image_and_kspace(selection):
    image, kspace = from_multicoil_train_file_to_image_and_kspace(test_file_multi_coil, selection=selection)
    reconstructed_image = crop_center(np.linalg.norm(ifft(kspace), axis=1), 320)
    np.testing.assert_allclose(reconstructed_image, image, rtol=1e-4, atol=1e-11)

@pytest.mark.skipif(not os.path.isfile(test_file_multi_coil), reason='test multi coil file not present for h5 utils.')
def test_from_multicoil_train_file_to_image_and_kspace_parallel():
    selection = [{'inner_slices': 8, 'rand': True}, {'rand': True, 'keep_dim': False}]
    _, kspace = from_multicoil_train_file_to_image_and_kspace(test_file_multi_coil, selection=selection)
    assert len(kspace.shape) == 3
