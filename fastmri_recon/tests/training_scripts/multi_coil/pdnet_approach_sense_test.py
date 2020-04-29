import pytest

from fastmri_recon.training_scripts.multi_coil import pdnet_approach_sense
from fastmri_recon.training_scripts.multi_coil.pdnet_approach_sense import train_pdnet

@pytest.mark.parametrize('args',[
    ('1', None, '0123', 2, 1, 2,),
])
def test_train_pdnet(create_full_fastmri_test_tmp_dataset, args):
    pdnet_approach_sense.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    pdnet_approach_sense.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    pdnet_approach_sense.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    pdnet_approach_sense.n_volumes_train = 2
    train_pdnet(*args)
    # TODO: checks that the checkpoints and the logs are correctly created
