import pytest

from fastmri_recon.training_scripts import ncpdnet_train
from fastmri_recon.training_scripts.unet_nc_train import train_unet


@pytest.mark.parametrize('kwargs',[
    {'dcomp': True},
])
def test_train_unet(create_full_fastmri_test_tmp_dataset, kwargs):
    ncpdnet_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    ncpdnet_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    ncpdnet_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    ncpdnet_train.n_volumes_train = 2
    train_unet(
        af=8,
        n_samples=2,
        n_epochs=1,
        base_n_filters=4,
        n_layers=2,
        **kwargs,
    )
    # TODO: checks that the checkpoints and the logs are correctly created
