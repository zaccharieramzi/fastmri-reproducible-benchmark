import os
import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.models.subclassed_models.pdnet import PDNet
from fastmri_recon.models.training.compile import default_model_compile


n_volumes_train = 973

def train_pdnet(af, contrast, cuda_visible_devices, n_samples, n_epochs, n_iter):
    # paths
    train_path = f'{FASTMRI_DATA_DIR}multicoil_train/'
    val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    af = int(af)
    # generators
    train_set = train_masked_kspace_dataset_from_indexable(
        train_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=True,
        scale_factor=1e6,
        n_samples=n_samples,
        parallel=False,
    )
    val_set = train_masked_kspace_dataset_from_indexable(
        val_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=True,
        scale_factor=1e6,
        parallel=False,
    )

    run_params = {
        'n_primal': 5,
        'n_dual': 1,
        'primal_only': True,
        'n_iter': n_iter,
        'multicoil': True,
        'n_filters': 32,
    }
    additional_info = f'af{af}'
    if contrast is not None:
        additional_info += f'_{contrast}'
    if n_samples is not None:
        additional_info += f'_{n_samples}'
    if n_iter != 10:
        additional_info += f'_i{n_iter}'

    run_id = f'pdnet_sense_{additional_info}_{int(time.time())}'
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

    model = PDNet(**run_params)
    default_model_compile(model, lr=1e-3)
    print(run_id)

    model.fit(
        train_set,
        steps_per_epoch=n_volumes_train,
        epochs=n_epochs,
        validation_data=val_set,
        validation_steps=2,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback,],
    )
    return run_id
