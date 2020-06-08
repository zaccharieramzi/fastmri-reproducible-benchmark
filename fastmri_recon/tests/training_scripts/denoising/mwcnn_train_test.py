from fastmri_recon.training_scripts.denoising import generic_train
from fastmri_recon.training_scripts.denoising.mwcnn_train import train_mwcnn


def test_train_mwcnn(create_full_fastmri_test_tmp_dataset):
    generic_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    generic_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    generic_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    generic_train.n_volumes_train = 2
    train_kwargs = dict(
        n_samples=2,
        n_epochs=1,
    )
    train_mwcnn(
        train_kwargs=train_kwargs,
        n_scales=3,
        kernel_size=3,
        bn=False,
        n_filters_per_scale=[4, 8, 8],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=2,
        first_conv_n_filters=4,
        res=True,
        n_outputs=1,
    )
