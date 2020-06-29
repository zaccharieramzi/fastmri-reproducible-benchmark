import numpy as np
import pytest
import tensorflow as tf

from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable


image_shape = [2, 640, 322, 1]
af = 4
us = af / (2 / np.pi)
image_size = [640, 474]
kspace_shape = [image_shape[0], 1, 640 * (474//af), 1]
file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape, orig_shape, use_af', [
    ({}, kspace_shape, image_shape[-2], True),
    ({'inner_slices': 1}, [1] + kspace_shape[1:], image_shape[-2], True),
    ({'inner_slices': 1, 'rand': True}, [1] + kspace_shape[1:], image_shape[-2], True),
    ({'contrast': file_contrast}, kspace_shape, image_shape[-2], True),
    ({'n_samples': 1}, kspace_shape, image_shape[-2], True),
    ({}, kspace_shape, image_shape[-2], False),
])
def test_train_nc_kspace_dataset_from_indexable(
        create_full_fastmri_test_tmp_dataset,
        ds_kwargs,
        expected_kspace_shape,
        orig_shape,
        use_af,
    ):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = train_nc_kspace_dataset_from_indexable(
        path,
        image_size,
        af=af if use_af else None,
        us=None if use_af else us,
        **ds_kwargs,
    )
    (kspace, traj, (shape,)), image = next(iter(ds))
    # shape verifications
    assert kspace.shape.as_list() == expected_kspace_shape
    assert shape.numpy()[0] == orig_shape
    assert traj.shape.as_list() == [expected_kspace_shape[0], 2, 640 * (474//af)]
    assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]

def test_spiral_dataset(create_full_fastmri_test_tmp_dataset):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = train_nc_kspace_dataset_from_indexable(
        path,
        image_size,
        af=af
    )
    (kspace, traj, (shape,)), image = next(iter(ds))
    # shape verifications
    assert kspace.shape.as_list() == kspace_shape
    assert shape.numpy()[0] == image_shape[-2]
    assert traj.shape.as_list() == [kspace_shape[0], 2, 640 * (474//af)]
    assert image.shape.as_list() == kspace_shape[0:1] + [320, 320, 1]
