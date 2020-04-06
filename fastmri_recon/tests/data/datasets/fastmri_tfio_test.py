import os

import pytest

from fastmri_recon.data.datasets.fastmri_tfio import train_masked_kspace_dataset_io
from fastmri_recon.models.subclassed_models.pdnet import PDNet
from fastmri_recon.models.training.compile import default_model_compile


test_file_single_coil = 'fastmri_recon/tests/fastmri_data/single_coil/file1000002.h5'
kspace_shape = [38, 640, 368, 1]
file_contrast = 'CORPDFS_FBK'

@pytest.mark.skipif(not os.path.isfile(test_file_single_coil), reason='test single coil file not present for single dataset.')
@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape', [
    ({}, kspace_shape),
    ({'inner_slices': 8}, [8,] + kspace_shape[1:]),
    ({'inner_slices': 8, 'rand': True}, [1,] + kspace_shape[1:]),
    # ({'contrast': file_contrast}, kspace_shape),  # TODO: implement this for tfio dataset
    # ({'n_samples': 1}, kspace_shape),  # TODO: implement this for tfio dataset
])
def test_train_masked_kspace_dataset_io(ds_kwargs, expected_kspace_shape):
    ds = train_masked_kspace_dataset_io('fastmri_recon/tests/fastmri_data/single_coil/', **ds_kwargs)
    (kspace, mask), image = next(iter(ds))
    assert kspace.shape.as_list() == expected_kspace_shape
    assert mask.shape.as_list() == expected_kspace_shape[:3]
    assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]

@pytest.mark.skipif(not os.path.isfile(test_file_single_coil), reason='test single coil file not present for single dataset.')
def test_train_masked_kspace_dataset_io_graph_mode():
    ds = train_masked_kspace_dataset_io('fastmri_recon/tests/fastmri_data/single_coil/', rand=True)
    model = PDNet(primal_only=True, n_iter=1, n_filters=8, n_primal=1, n_dual=1)
    default_model_compile(model, lr=1e-3)
    model.fit(ds, steps_per_epoch=1, epochs=2)
