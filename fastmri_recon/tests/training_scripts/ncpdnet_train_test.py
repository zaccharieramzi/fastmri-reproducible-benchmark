import pytest
import os

from fastmri_recon.training_scripts import nc_train
from fastmri_recon.training_scripts.nc_train import train_ncpdnet


@pytest.mark.parametrize('kwargs',[
    {'n_iter': 1},
    {'n_iter': 2},
    {'dcomp': True},
])
@pytest.mark.parametrize('nufft_implementation',['tfkbnufft', 'tensorflow-nufft'])
def test_train_ncpdnet(create_full_fastmri_test_tmp_dataset, kwargs, nufft_implementation):
    nc_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    nc_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    nc_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    nc_train.n_volumes_train_fastmri = 2
    if nufft_implementation == 'tensorflow-nufft' and os.environ.get('CI', False) == "True":
        pytest.skip('Skipping tensorflow-nufft CI test as it needs a GPU')
    train_ncpdnet(
        af=8,
        n_samples=2,
        n_epochs=1,
        n_filters=8,
        n_primal=2,
        nufft_implementation=nufft_implementation,
        **kwargs,
    )
    # TODO: checks that the checkpoints and the logs are correctly created

@pytest.mark.skipif(os.environ.get('CI', False) == "True", reason='Non cartesian multicoil is too long to run in CI.')
@pytest.mark.skip(reason='Currently this test is invalid because multicoil now relies on tfrecords')
def test_train_ncpdnet_multicoil(create_full_fastmri_test_tmp_dataset):
    nc_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    nc_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    nc_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    nc_train.n_volumes_train_fastmri = 1
    train_ncpdnet(
        af=8,
        n_samples=1,
        n_epochs=1,
        n_filters=4,
        n_primal=2,
        multicoil=True,
        dcomp=True,
    )
