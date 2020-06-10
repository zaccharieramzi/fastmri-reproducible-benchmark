import os.path as op

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc_denoising import train_noisy_dataset_from_indexable
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import build_model_from_specs
from fastmri_recon.models.training.compile import default_model_compile


n_volumes_train = 973

def train_denoiser(
        model,
        run_id,
        noise_std=30,
        contrast=None,
        n_samples=None,
        n_epochs=200,
        loss='mae',
        lr=1e-4,
    ):
    train_path = f'{FASTMRI_DATA_DIR}singlecoil_train/singlecoil_train/'
    val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'
    train_set = train_noisy_dataset_from_indexable(
        train_path,
        noise_std=noise_std,
        contrast=contrast,
        inner_slices=None,
        rand=True,
        scale_factor=1e6,
        n_samples=n_samples,
    )
    val_set = train_noisy_dataset_from_indexable(
        val_path,
        noise_std=noise_std,
        contrast=contrast,
        inner_slices=None,
        rand=True,
        scale_factor=1e6,
    )
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
    tqdm_cback = TQDMProgressBar()
    if isinstance(model, tuple):
        model = build_model_from_specs(*model)
    default_model_compile(model, lr=lr, loss=loss)
    model.fit(
        train_set,
        steps_per_epoch=n_volumes_train,
        epochs=n_epochs,
        validation_data=val_set,
        validation_steps=2,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback, tqdm_cback],
    )
    return run_id
