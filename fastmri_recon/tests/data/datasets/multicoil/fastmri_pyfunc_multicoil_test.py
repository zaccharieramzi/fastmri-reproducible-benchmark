import os

import pytest

from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable


test_file_multi_coil = 'fastmri_recon/tests/fastmri_data/multi_coil/file1000001.h5'
kspace_shape = [36, 15, 640, 372, 1]
file_contrast = 'CORPDFS_FBK'

@pytest.mark.skipif(not os.path.isfile(test_file_multi_coil), reason='test multi coil file not present for multi coil dataset.')
@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape', [
    ({}, kspace_shape[0:1] + kspace_shape[2:]),
    ({'parallel': False}, kspace_shape),
    ({'inner_slices': 8}, [8,] + kspace_shape[2:]),
    ({'inner_slices': 8, 'rand': True}, [1,] + kspace_shape[2:]),
    ({'contrast': file_contrast, 'parallel': False}, kspace_shape),
    ({'n_samples': 1, 'parallel': False}, kspace_shape)
])
def test_train_masked_kspace_dataset_from_indexable(ds_kwargs, expected_kspace_shape):
    ds = train_masked_kspace_dataset_from_indexable('fastmri_recon/tests/fastmri_data/multi_coil/', **ds_kwargs)
    (kspace, mask), image = next(iter(ds))
    assert kspace.shape.as_list() == expected_kspace_shape
    assert mask.shape.as_list() == expected_kspace_shape[:-1]
    if ds_kwargs.get('parallel', True):
        assert image.shape.as_list() == expected_kspace_shape
    else:
        assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]