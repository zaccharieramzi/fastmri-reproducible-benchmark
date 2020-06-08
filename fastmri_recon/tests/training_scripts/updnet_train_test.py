import pytest

from fastmri_recon.training_scripts import updnet_train
from fastmri_recon.training_scripts.updnet_train import train_updnet

@pytest.mark.parametrize('kwargs',[
    {'use_mixed_precision': True},
    {'loss': 'compound_mssim'},
    {'non_linearity': 'prelu'},
    {'non_linearity': 'prelu', 'refine_smaps': True},
    {'fixed_masks': True},
    {'multicoil': False},
])
def test_train_updnet(create_full_fastmri_test_tmp_dataset, kwargs):
    updnet_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    updnet_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    updnet_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    updnet_train.n_volumes_train = 2
    train_updnet(
        af=1,
        n_samples=2,
        n_epochs=1,
        n_iter=1,
        n_layers=2,
        base_n_filter=2,
        **kwargs,
    )
    # TODO: checks that the checkpoints and the logs are correctly created
