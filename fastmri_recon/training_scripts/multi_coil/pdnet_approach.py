import os
import os.path as op
import time

import click
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.models.functional_models.pdnet import pdnet



# paths
train_path = f'{FASTMRI_DATA_DIR}multicoil_train/'
val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'

n_volumes_train = 973

@click.command()
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
    default=None,
    type=click.Choice(['CORPDFS_FBK', 'CORPD_FBK', None], case_sensitive=False),
    help='The contrast chosen for this fine-tuning. Defaults to None.',
)
@click.option(
    'cuda_visible_devices',
    '-gpus',
    '--cuda-visible-devices',
    default='0123',
    type=str,
    help='The visible GPU devices. Defaults to 0123',
)
@click.option(
    'n_samples',
    '-ns',
    default=None,
    type=int,
    help='The number of samples to use for this training. Default to None, which means all samples are used.',
)
@click.option(
    'n_epochs',
    '-e',
    default=300,
    type=int,
    help='The number of epochs to train the model. Default to 300.',
)
def train_pdnet(af, contrast, cuda_visible_devices, n_samples, n_epochs):
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
        n_samples=n_samples,
    )
    val_set = train_masked_kspace_dataset_from_indexable(
        val_path,
        AF=af,
        contrast=contrast,
        inner_slices=8,
        rand=True,
        scale_factor=1e6,
    )

    run_params = {
        'n_primal': 5,
        'n_dual': 5,
        'n_iter': 10,
        'n_filters': 32,
        'fastmri': False,
    }
    additional_info = f'af{af}'
    if contrast is not None:
        additional_info += f'_{contrast}'
    if n_samples is not None:
        additional_info += f'_{n_samples}'

    run_id = f'pdnet_multicoil_{additional_info}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'

    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=True)
    log_dir = op.join(f'{LOGS_DIR}logs', run_id)
    tboard_cback = TensorBoard(
        profile_batch=0,
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
    )

    model = pdnet(lr=1e-3, **run_params)
    print(run_id)
    print(model.summary(line_length=150))
    model.fit(
        train_set,
        steps_per_epoch=n_volumes_train,
        epochs=n_epochs,
        validation_data=val_set,
        validation_steps=5,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback,],
    )

if __name__ == '__main__':
    train_pdnet()
