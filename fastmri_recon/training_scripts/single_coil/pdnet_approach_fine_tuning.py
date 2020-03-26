import os
import os.path as op
import time

import click
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.models.functional_models.pdnet import pdnet



# paths
train_path = f'{FASTMRI_DATA_DIR}singlecoil_train/singlecoil_train/'
val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'
test_path = f'{FASTMRI_DATA_DIR}singlecoil_test/'

n_volumes_train = 973

@click.command()
@click.argument('original_run_id')
@click.option(
    'af',
    '-a',
    default='4',
    type=click.Choice(['4', '8']),
    help='The acceleration factor chosen for this fine tuning. Defaults to 4.',
)
@click.option(
    'contrast',
    '-c',
    default='CORPDFS_FBK',
    type=click.Choice(['CORPDFS_FBK', 'CORPD_FBK'], case_sensitive=False),
    help='The contrast chosen for this fine-tuning. Defaults to CORPDFS_FBK.',
)
@click.option(
    'cuda_visible_devices',
    '-gpus',
    '--cuda-visible-devices',
    default='0123',
    type=str,
    help='The visible GPU devices. Defaults to 0123',
)
def fine_tune_pdnet(original_run_id, af, contrast, cuda_visible_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    af = int(af)
    # generators
    train_set = train_masked_kspace_dataset_from_indexable(
        train_path,
        AF=af,
        contrast=contrast,
        inner_slices=8,
        rand=True,
        scale_factor=1e6,
    )
    val_set = train_masked_kspace_dataset_from_indexable(
        val_path,
        AF=af,
        contrast=contrast,
        scale_factor=1e6,
    )

    run_params = {
        'n_primal': 5,
        'n_dual': 5,
        'n_iter': 10,
        'n_filters': 32,
    }
    n_epochs = 50
    new_run_id = f'pdnet_af{af}_{contrast}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{new_run_id}' + '-{epoch:02d}.hdf5'

    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=True)
    log_dir = op.join(f'{LOGS_DIR}logs', new_run_id)
    tboard_cback = TensorBoard(
        profile_batch=0,
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
    )

    model = pdnet(lr=1e-6, **run_params)
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{original_run_id}-300.hdf5')
    print(model.summary(line_length=150))
    model.fit(
        train_set,
        steps_per_epoch=n_volumes_train//2,
        epochs=n_epochs,
        validation_data=val_set,
        validation_steps=5,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback,],
    )

if __name__ == '__main__':
    fine_tune_pdnet()
