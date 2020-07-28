import os
import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
from fastmri_recon.models.subclassed_models.unet import UnetComplex
from fastmri_recon.models.training.compile import default_model_compile


n_volumes_train = 973
# this number means that 99.56% of all images will not be affected by
# cropping
IM_SIZE = (640, 400)

def train_ncnet(
        model,
        run_id=None,
        multicoil=False,
        acq_type='radial',
        dcomp=False,
        contrast=None,
        cuda_visible_devices='0123',
        n_samples=None,
        n_epochs=200,
        use_mixed_precision=False,
        loss='mae',
        original_run_id=None,
        **acq_kwargs,
    ):
    # paths
    if multicoil:
        train_path = f'{FASTMRI_DATA_DIR}multicoil_train/'
        val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
    else:
        train_path = f'{FASTMRI_DATA_DIR}singlecoil_train/singlecoil_train/'
        val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'


    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    # trying mixed precision
    if use_mixed_precision:
        policy_type = 'mixed_float16'
    else:
        policy_type = 'float32'
    policy = mixed_precision.Policy(policy_type)
    mixed_precision.set_policy(policy)
    # generators
    if multicoil:
        dataset = multicoil_dataset
    else:
        dataset = singlecoil_dataset
    train_set = dataset(
        train_path,
        IM_SIZE,
        acq_type=acq_type,
        compute_dcomp=dcomp,
        contrast=contrast,
        inner_slices=None,
        rand=True,
        scale_factor=1e6,
        n_samples=n_samples,
        **acq_kwargs
    )
    val_set = dataset(
        val_path,
        IM_SIZE,
        acq_type=acq_type,
        compute_dcomp=dcomp,
        contrast=contrast,
        inner_slices=None,
        rand=True,
        scale_factor=1e6,
        **acq_kwargs
    )

    additional_info = f'{acq_type}'
    if contrast is not None:
        additional_info += f'_{contrast}'
    if n_samples is not None:
        additional_info += f'_{n_samples}'
    if loss != 'mae':
        additional_info += f'_{loss}'
    if dcomp:
        additional_info += '_dcomp'
    run_id = f'{run_id}_{additional_info}_{int(time.time())}'
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

    if original_run_id is not None:
        lr = 1e-7
        n_steps = n_volumes_train//2
    else:
        lr = 1e-4
        n_steps = n_volumes_train
    default_model_compile(model, lr=lr, loss=loss)
    print(run_id)
    if original_run_id is not None:
        if os.environ.get('FASTMRI_DEBUG'):
            n_epochs_original = 1
        else:
            n_epochs_original = 250
        model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{original_run_id}-{n_epochs_original:02d}.hdf5')

    model.fit(
        train_set,
        steps_per_epoch=n_steps,
        epochs=n_epochs,
        validation_data=val_set,
        validation_steps=2,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback, tqdm_cback],
    )
    return run_id

def train_ncpdnet(
        multicoil=False,
        dcomp=False,
        normalize_image=False,
        n_iter=10,
        n_filters=32,
        n_primal=5,
        non_linearity='relu',
        **train_kwargs,
    ):
    run_params = {
        'n_primal': n_primal,
        'multicoil': multicoil,
        'activation': non_linearity,
        'n_iter': n_iter,
        'n_filters': n_filters,
        'im_size': IM_SIZE,
        'dcomp': dcomp,
        'normalize_image': normalize_image,
    }

    if multicoil:
        ncpdnet_type = 'ncpdnet_sense_'
    else:
        ncpdnet_type = 'ncpdnet_singlecoil_'
    additional_info = ''
    if n_iter != 10:
        additional_info += f'_i{n_iter}'
    if non_linearity != 'relu':
        additional_info += f'_{non_linearity}'


    run_id = f'{ncpdnet_type}_{additional_info}'
    model = NCPDNet(**run_params)

    train_ncnet(
        model,
        run_id=run_id,
        multicoil=multicoil,
        dcomp=dcomp,
        **train_kwargs,
    )

def train_unet_nc(
        multicoil=False,
        dcomp=False,
        n_layers=4,
        base_n_filters=16,
        non_linearity='relu',
        **train_kwargs,
    ):
    run_params = {
        'non_linearity': non_linearity,
        'n_layers': n_layers,
        'layers_n_channels': [base_n_filters * 2**i for i in range(n_layers)],
        'layers_n_non_lins': 2,
        'res': True,
        'im_size': IM_SIZE,
        'dcomp': dcomp,
        'dealiasing_nc_fastmri': True,
        'multicoil': multicoil,
    }

    if multicoil:
        unet_type = 'unet_mc_'
    else:
        unet_type = 'unet_singlecoil_'
    additional_info = ''
    if non_linearity != 'relu':
        additional_info += f'_{non_linearity}'

    run_id = f'{unet_type}_{additional_info}'

    model = UnetComplex(**run_params)
    train_ncnet(
        model,
        run_id=run_id,
        multicoil=multicoil,
        dcomp=dcomp,
        **train_kwargs,
    )
