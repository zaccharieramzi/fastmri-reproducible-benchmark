import pytest

from fastmri_recon.training_scripts.multi_coil import updnet_approach_sense
from fastmri_recon.training_scripts.multi_coil.updnet_approach_sense import train_updnet

@pytest.mark.parametrize('args',[
    ('1', None, '0123', 2, 1, 2, True, 2, 2),
    ('1', None, '0123', 2, 1, 2, False, 2, 2),
    ('1', None, '0123', 2, 1, 2, False, 2, 2, 'relu', 'compound_mssim'),
    ('1', None, '0123', 2, 1, 2, False, 2, 2, 'prelu', 'compound_mssim'),
])
def test_train_updnet(create_full_fastmri_test_tmp_dataset, args):
    updnet_approach_sense.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    updnet_approach_sense.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    updnet_approach_sense.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    updnet_approach_sense.n_volumes_train = 2
    train_updnet(*args)
    # TODO: checks that the checkpoints and the logs are correctly created
