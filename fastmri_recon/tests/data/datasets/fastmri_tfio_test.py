import os

import pytest

from fastmri_recon.data.datasets.fastmri_tfio import train_masked_kspace_dataset_io
from fastmri_recon.models.subclassed_models.pdnet import PDNet
from fastmri_recon.models.training.compile import default_model_compile


kspace_shape = [2, 640, 322, 1]
file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape', [
    ({}, kspace_shape),
    ({'inner_slices': 1}, [1,] + kspace_shape[1:]),
    ({'inner_slices': 1, 'rand': True}, [1,] + kspace_shape[1:]),
    # ({'contrast': file_contrast}, kspace_shape),  # TODO: implement this for tfio dataset
    # ({'n_samples': 1}, kspace_shape),  # TODO: implement this for tfio dataset
])
def test_train_masked_kspace_dataset_io(create_full_fastmri_test_tmp_dataset, ds_kwargs, expected_kspace_shape):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = train_masked_kspace_dataset_io(path, **ds_kwargs)
    (kspace, mask), image = next(iter(ds))
    assert kspace.shape.as_list() == expected_kspace_shape
    assert mask.shape.as_list() == [1 for _ in expected_kspace_shape[:-2]] + [expected_kspace_shape[-2]]
    assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]

def test_train_masked_kspace_dataset_io_graph_mode(create_full_fastmri_test_tmp_dataset):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = train_masked_kspace_dataset_io(path, rand=True)
    model = PDNet(primal_only=True, n_iter=1, n_filters=8, n_primal=1, n_dual=1)
    default_model_compile(model, lr=1e-3)
    model.fit(ds, steps_per_epoch=1, epochs=2)
