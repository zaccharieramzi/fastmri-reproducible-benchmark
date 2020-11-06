import os.path as op

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow_addons.callbacks import TQDMProgressBar
from tf_fastmri_data.datasets.noisy import NoisyFastMRIDatasetBuilder

from fastmri_recon.config import *
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import build_model_from_specs
from fastmri_recon.models.training.compile import default_model_compile


def train_denoiser(
        model,
        run_id,
        noise_std=30,
        contrast=None,
        n_samples=None,
        n_epochs=200,
        loss='mae',
        lr=1e-4,
        n_steps_per_epoch=973,  # number of volumes in the fastMRI dataset
    ):
    ds_kwargs = dict(
        contrast=contrast,
        slice_random=True,
        scale_factor=1e4,
        noise_input=False,
        noise_power_spec=noise_std,
        noise_mode='gaussian',
    )
    train_set = NoisyFastMRIDatasetBuilder(
        dataset='train',
        n_samples=n_samples,
        **ds_kwargs,
    ).preprocessed_ds
    val_set = NoisyFastMRIDatasetBuilder(
        dataset='val',
        **ds_kwargs,
    ).preprocessed_ds
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
        steps_per_epoch=n_steps_per_epoch,
        epochs=n_epochs,
        validation_data=val_set,
        validation_steps=100,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback, tqdm_cback],
    )
    return run_id
