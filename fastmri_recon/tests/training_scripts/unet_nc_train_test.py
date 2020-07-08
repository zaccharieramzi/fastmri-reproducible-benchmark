import pytest

from fastmri_recon.training_scripts import nc_train
from fastmri_recon.training_scripts.nc_train import train_unet_nc


@pytest.mark.parametrize('kwargs',[
    {'dcomp': True},
])
def test_train_unet_nc(create_full_fastmri_test_tmp_dataset, kwargs):
    nc_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    nc_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    nc_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    nc_train.n_volumes_train = 2
    train_unet_nc(
        af=8,
        n_samples=2,
        n_epochs=1,
        base_n_filters=4,
        n_layers=2,
        **kwargs,
    )
    # TODO: checks that the checkpoints and the logs are correctly created
