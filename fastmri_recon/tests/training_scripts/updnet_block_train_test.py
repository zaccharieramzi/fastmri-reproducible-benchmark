from fastmri_recon.training_scripts import updnet_block_train
from fastmri_recon.training_scripts.updnet_block_train import train_updnet_block


def test_train_updnet(create_full_fastmri_test_tmp_dataset):
    updnet_block_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    updnet_block_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    updnet_block_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    updnet_block_train.n_volumes_train = 2
    train_updnet_block(
        af=4,
        n_samples=2,
        n_epochs_per_block=2,
        n_iter=2,
        n_layers=2,
        base_n_filter=2,
    )
    # TODO: checks that the checkpoints and the logs are correctly created
