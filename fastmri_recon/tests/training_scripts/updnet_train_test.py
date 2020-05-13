import pytest

from fastmri_recon.training_scripts import updnet_train
from fastmri_recon.training_scripts.updnet_train import train_updnet

@pytest.mark.parametrize('args',[
    ('1', None, '0123', 2, 1, 2, True, 2, 2),
    ('1', None, '0123', 2, 1, 2, False, 2, 2),
    ('1', None, '0123', 2, 1, 2, False, 2, 2, 'relu', None, False, 'compound_mssim'),
    ('1', None, '0123', 2, 1, 2, False, 2, 2, 'prelu', None, False, 'mae'),
    ('1', None, '0123', 2, 1, 2, False, 2, 2, 'prelu', None, True),
])
def test_train_updnet(create_full_fastmri_test_tmp_dataset, args):
    updnet_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    updnet_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    updnet_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    updnet_train.n_volumes_train = 2
    train_updnet(*args)
    # TODO: checks that the checkpoints and the logs are correctly created
