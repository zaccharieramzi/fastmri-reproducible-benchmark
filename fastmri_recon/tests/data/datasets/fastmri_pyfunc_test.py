import os

import pytest

from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable


test_file_single_coil = 'fastmri_recon/tests/fastmri_data/single_coil/file1000002.h5'
kspace_shape = [38, 640, 368, 1]
file_contrast = 'CORPDFS_FBK'

@pytest.mark.skipif(not os.path.isfile(test_file_single_coil), reason='test single coil file not present for single dataset.')
@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape', [
    ({}, kspace_shape),
    ({'inner_slices': 8}, [8,] + kspace_shape[1:]),
    ({'inner_slices': 8, 'rand': True}, [1,] + kspace_shape[1:]),
    ({'contrast': file_contrast}, kspace_shape),
    ({'n_samples': 1}, kspace_shape)
])
def test_train_masked_kspace_dataset_from_indexable(ds_kwargs, expected_kspace_shape):
    ds = train_masked_kspace_dataset_from_indexable('fastmri_recon/tests/fastmri_data/single_coil/', **ds_kwargs)
    (kspace, mask), image = next(iter(ds))
    assert kspace.shape.as_list() == expected_kspace_shape
    assert mask.shape.as_list() == expected_kspace_shape[:3]
    assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]
