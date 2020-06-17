import pytest
import tensorflow as tf

from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable


image_shape = [2, 640, 322, 1]
nspokes = 15
spokelength = 400
image_size = [640, 474]
file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape, orig_shape', [
    ({}, [image_shape[0], 1, nspokes*spokelength], image_shape[-2]),
    ({'inner_slices': 1}, [1, 1, nspokes*spokelength], image_shape[-2]),
    ({'inner_slices': 1, 'rand': True}, [1, 1, nspokes*spokelength], image_shape[-2]),
    ({'contrast': file_contrast}, [image_shape[0], 1, nspokes*spokelength], image_shape[-2]),
    ({'n_samples': 1}, [image_shape[0], 1, nspokes*spokelength], image_shape[-2]),
])
def test_train_nc_kspace_dataset_from_indexable(
        create_full_fastmri_test_tmp_dataset,
        ds_kwargs,
        expected_kspace_shape,
        orig_shape,
    ):
    tf.config.experimental_run_functions_eagerly(True)
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = train_nc_kspace_dataset_from_indexable(
        path,
        image_size,
        nspokes=nspokes,
        spokelength=spokelength,
        **ds_kwargs,
    )
    (kspace, traj, (shape,)), image = next(iter(ds))
    # shape verifications
    assert kspace.shape.as_list() == expected_kspace_shape
    assert shape.numpy()[0] == orig_shape
    assert traj.shape.as_list() == [expected_kspace_shape[0], 2, nspokes * spokelength]
    assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]
