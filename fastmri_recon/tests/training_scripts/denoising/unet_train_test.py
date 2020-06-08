from fastmri_recon.training_scripts.denoising import generic_train
from fastmri_recon.training_scripts.denoising.unet_train import train_unet


def test_train_unet(create_full_fastmri_test_tmp_dataset):
    generic_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    generic_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    generic_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    generic_train.n_volumes_train = 2
    train_kwargs = dict(
        n_samples=2,
        n_epochs=1,
    )
    run_params = {
        'n_layers': 2,
        'pool': 'max',
        "layers_n_channels": [16, 32,],
        'layers_n_non_lins': 1,
        'res': True,
        'input_size': (None, None, 1),
    }
    train_unet(
        train_kwargs=train_kwargs,
        **run_params,
    )
