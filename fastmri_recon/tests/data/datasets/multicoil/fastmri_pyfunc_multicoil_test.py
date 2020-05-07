import os

import pytest
import tensorflow as tf

from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable


kspace_shape = [2, 15, 640, 322, 1]
file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape', [
    ({}, kspace_shape[0:1] + kspace_shape[2:]),
    ({'parallel': False}, kspace_shape),
    ({'inner_slices': 1}, [1,] + kspace_shape[2:]),
    ({'inner_slices': 1, 'rand': True}, [1,] + kspace_shape[2:]),
    ({'inner_slices': 1, 'rand': True, 'parallel': False}, [1,] + kspace_shape[1:]),
    ({'contrast': file_contrast, 'parallel': False}, kspace_shape),
    ({'n_samples': 1, 'parallel': False}, kspace_shape)
])
def test_train_masked_kspace_dataset_from_indexable(create_full_fastmri_test_tmp_dataset, ds_kwargs, expected_kspace_shape):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_train']
    ds = train_masked_kspace_dataset_from_indexable(path, AF=1, **ds_kwargs)
    if ds_kwargs.get('parallel', True):
        (kspace, mask), image = next(iter(ds))
    else:
        (kspace, mask, smaps), image = next(iter(ds))
    # shape verifications
    assert kspace.shape.as_list() == expected_kspace_shape
    assert mask.shape.as_list() == [expected_kspace_shape[0]] + [1 for _ in expected_kspace_shape[1:-2]] + [expected_kspace_shape[-2]]
    if ds_kwargs.get('parallel', True):
        assert image.shape.as_list() == expected_kspace_shape
    else:
        assert smaps.shape.as_list() == expected_kspace_shape[:-1]
        assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]
    # content verifications
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllInSet(mask, [1 + 0.j])  # this because we set af to 1
    # TODO: implement adjoint fourier multicoil
