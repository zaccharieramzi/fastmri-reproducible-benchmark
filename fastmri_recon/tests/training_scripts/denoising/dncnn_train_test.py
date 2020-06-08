from fastmri_recon.training_scripts.denoising import generic_train
from fastmri_recon.training_scripts.denoising.dncnn_train import train_dncnn


def test_train_updnet(create_full_fastmri_test_tmp_dataset):
    generic_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    generic_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    generic_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    generic_train.n_volumes_train = 2
    train_kwargs = dict(
        n_samples=2,
        n_epochs=1,
    )
    train_dncnn(
        train_kwargs=train_kwargs,
        n_convs=2,
        n_filters=4,
        res=True,
        n_outputs=1,
    )
