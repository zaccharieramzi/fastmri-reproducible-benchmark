import os
import os.path as op
from pathlib import Path
import time

import click
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.data.datasets.oasis_tf_records import train_nc_kspace_dataset_from_tfrecords as three_d_dataset
from fastmri_recon.data.datasets.multicoil.non_cartesian_tf_records import train_nc_kspace_dataset_from_tfrecords as multicoil_dataset
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
from fastmri_recon.models.subclassed_models.pdnet import PDNet
from fastmri_recon.models.subclassed_models.unet import UnetComplex
from fastmri_recon.models.subclassed_models.vnet import VnetComplex
from fastmri_recon.models.training.compile import default_model_compile
from fastmri_recon.training_scripts.custom_objects import CUSTOM_TF_OBJECTS
from fastmri_recon.training_scripts.model_saving_workaround import ModelCheckpointWorkAround


n_volumes_train_fastmri = 973
n_volumes_train_oasis = 3273
# this number means that 99.56% of all images will not be affected by
# cropping
IM_SIZE = (640, 400)
VOLUME_SIZE = (176, 256, 256)

def train_ncnet(
        model,
        run_id=None,
        multicoil=False,
        three_d=False,
        acq_type='radial',
        scale_factor=1e6,
        dcomp=False,
        contrast=None,
        cuda_visible_devices='0123',
        n_samples=None,
        n_epochs=200,
        use_mixed_precision=False,
        loss='mae',
        original_run_id=None,
        checkpoint_epoch=0,
        save_state=False,
        lr=1e-4,
        **acq_kwargs,
    ):
    # paths
    n_volumes_train = n_volumes_train_fastmri
    if multicoil:
        train_path = f'{FASTMRI_DATA_DIR}multicoil_train/'
        val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
    elif three_d:
        train_path = f'{OASIS_DATA_DIR}/train/'
        val_path = f'{OASIS_DATA_DIR}/val/'
        n_volumes_train = n_volumes_train_oasis
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
    mixed_precision.set_global_policy(policy)
    # generators
    if multicoil:
        dataset = multicoil_dataset
        image_size = IM_SIZE
    elif three_d:
        dataset = three_d_dataset
        image_size = VOLUME_SIZE
    else:
        dataset = singlecoil_dataset
        image_size = IM_SIZE
    if not three_d:
        add_kwargs = {
            'contrast': contrast,
            'rand': True,
            'inner_slices': None,
        }
    else:
        add_kwargs = {}
    add_kwargs.update(**acq_kwargs)
    train_set = dataset(
        train_path,
        image_size,
        acq_type=acq_type,
        compute_dcomp=dcomp,
        scale_factor=scale_factor,
        n_samples=n_samples,
        **add_kwargs
    )
    val_set = dataset(
        val_path,
        image_size,
        acq_type=acq_type,
        compute_dcomp=dcomp,
        scale_factor=scale_factor,
        **add_kwargs
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
    if checkpoint_epoch == 0:
        run_id = f'{run_id}_{additional_info}_{int(time.time())}'
    else:
        run_id = original_run_id
    final_epoch = checkpoint_epoch + n_epochs
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'

    log_dir = op.join(f'{LOGS_DIR}logs', run_id)
    tboard_cback = TensorBoard(
        profile_batch=0,
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
    )
    tqdm_cback = TQDMProgressBar()

    n_steps = n_volumes_train

    chkpt_cback = ModelCheckpointWorkAround(
        chkpt_path,
        save_freq=int(n_epochs*n_steps),
        save_weights_only=True,
    )
    default_model_compile(model, lr=lr, loss=loss)
    # first run of the model to avoid the saving error
    # ValueError: as_list() is not defined on an unknown TensorShape.
    # it can also allow loading of weights
    model(next(iter(train_set))[0])
    if not checkpoint_epoch == 0:
        model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{original_run_id}-{checkpoint_epoch:02d}.hdf5')
        grad_vars = model.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
        with open(f'{CHECKPOINTS_DIR}checkpoints/{original_run_id}-optimizer.pkl', 'rb') as f:
            weight_values = pickle.load(f)
        model.optimizer.set_weights(weight_values)
    print(run_id)


    model.fit(
        train_set,
        steps_per_epoch=n_steps,
        initial_epoch=checkpoint_epoch,
        epochs=final_epoch,
        validation_data=val_set,
        validation_steps=2,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback, tqdm_cback],
    )
    if save_state:
        symbolic_weights = getattr(model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-optimizer.pkl', 'wb') as f:
            pickle.dump(weight_values, f)
    return run_id

def train_ncpdnet(
        multicoil=False,
        three_d=False,
        dcomp=False,
        normalize_image=False,
        n_iter=10,
        n_filters=32,
        n_primal=5,
        non_linearity='relu',
        refine_smaps=True,
        nufft_implementation='tfkbnufft',
        **train_kwargs,
    ):
    if three_d:
        image_size = VOLUME_SIZE
    else:
        image_size = IM_SIZE
    run_params = {
        'n_primal': n_primal,
        'multicoil': multicoil,
        'three_d': three_d,
        'activation': non_linearity,
        'n_iter': n_iter,
        'n_filters': n_filters,
        'im_size': image_size,
        'dcomp': dcomp,
        'normalize_image': normalize_image,
        'refine_smaps': refine_smaps,
        'fastmri': not three_d,
        'nufft_implementation': nufft_implementation,
    }

    if multicoil:
        ncpdnet_type = 'ncpdnet_sense_'
    elif three_d:
        ncpdnet_type = 'ncpdnet_3d_'
    else:
        ncpdnet_type = 'ncpdnet_singlecoil_'
    additional_info = ''
    if n_iter != 10:
        additional_info += f'_i{n_iter}'
    if non_linearity != 'relu':
        additional_info += f'_{non_linearity}'
    if multicoil and refine_smaps:
        additional_info += '_rfs'


    run_id = f'{ncpdnet_type}_{additional_info}'
    model = NCPDNet(**run_params)

    return train_ncnet(
        model,
        run_id=run_id,
        multicoil=multicoil,
        dcomp=dcomp,
        three_d=three_d,
        **train_kwargs,
    )

def train_gridded_pdnet(
        n_iter=10,
        n_filters=32,
        n_primal=5,
        non_linearity='relu',
        **train_kwargs,
    ):
    run_params = {
        'n_primal': n_primal,
        'activation': non_linearity,
        'n_iter': n_iter,
        'n_filters': n_filters,
        'primal_only': True,
    }

    ncpdnet_type = 'pdnet_gridded_singlecoil_'
    additional_info = ''
    if n_iter != 10:
        additional_info += f'_i{n_iter}'
    if non_linearity != 'relu':
        additional_info += f'_{non_linearity}'


    run_id = f'{ncpdnet_type}_{additional_info}'
    model = PDNet(**run_params)
    train_kwargs.update(dict(
        multicoil=False,
        dcomp=False,
        three_d=False,
        gridding=True,
    ))
    return train_ncnet(
        model,
        run_id=run_id,
        **train_kwargs,
    )

def train_unet_nc(
        multicoil=False,
        dcomp=False,
        n_layers=4,
        base_n_filters=16,
        non_linearity='relu',
        nufft_implementation='tfkbnufft',
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
        'nufft_implementation': nufft_implementation,
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
    return train_ncnet(
        model,
        run_id=run_id,
        multicoil=multicoil,
        dcomp=dcomp,
        **train_kwargs,
    )

def train_vnet_nc(
        n_layers=4,
        dcomp=False,
        base_n_filters=16,
        non_linearity='relu',
        **train_kwargs,
    ):
    run_params = {
        'non_linearity': non_linearity,
        'layers_n_channels': [base_n_filters * 2**i for i in range(n_layers)],
        'layers_n_non_lins': 2,
        'res': True,
        'im_size': VOLUME_SIZE,
        'dcomp': dcomp,
        'dealiasing_nc': True,
    }

    vnet_type = 'vnet_3d_'
    additional_info = ''
    if non_linearity != 'relu':
        additional_info += f'_{non_linearity}'

    run_id = f'{vnet_type}_{additional_info}'

    model = VnetComplex(**run_params)
    train_kwargs.update(three_d=True)
    return train_ncnet(
        model,
        run_id=run_id,
        dcomp=dcomp,
        **train_kwargs,
    )

def train_ncnet_multinet(
        af=4,
        n_epochs=200,
        loss='mae',
        refine_smaps=False,
        multicoil=False,
        model='pdnet',
        acq_type='radial',
        scale_factor=1e6,
        three_d=False,
        dcomp=False,
        n_filters=None,
        n_iter=10,
        normalize_image=False,
        n_primal=5,
        use_mixed_precision=False,
        original_run_id=None,
        checkpoint_epoch=0,
        save_state=False,
    ):
    if model == 'pdnet':
        train_function = train_ncpdnet
        if n_filters is None:
            n_filters = 32
        add_kwargs = {
            'refine_smaps': refine_smaps,
            'n_filters': n_filters,
            'n_iter': n_iter,
            'normalize_image': normalize_image,
            'n_primal': n_primal,
        }
    elif model == 'pdnet-gridded':
        train_function = train_gridded_pdnet
        if n_filters is None:
            n_filters = 32
        add_kwargs = {
            'n_filters': n_filters,
            'n_iter': n_iter,
            'n_primal': n_primal,
        }
    elif model == 'unet':
        if n_filters is None:
            base_n_filters = 16
        else:
            base_n_filters = n_filters
        if three_d:
            train_function = train_vnet_nc
        else:
            train_function = train_unet_nc
        add_kwargs = {'base_n_filters': base_n_filters}
    if multicoil:
        add_kwargs.update(dcomp=True)
    else:
        add_kwargs.update(dcomp=dcomp)
    return train_function(
        af=af,
        n_epochs=n_epochs,
        loss=loss,
        multicoil=multicoil,
        acq_type=acq_type,
        scale_factor=scale_factor,
        three_d=three_d,
        use_mixed_precision=use_mixed_precision,
        original_run_id=original_run_id,
        checkpoint_epoch=checkpoint_epoch,
        save_state=save_state,
        **add_kwargs,
    )


@click.command()
@click.option(
    'af',
    '-a',
    type=int,
    default=4,
    help='The acceleration factor.'
)
@click.option(
    'n_epochs',
    '-e',
    type=int,
    default=70,
    help='The number of epochs.'
)
@click.option(
    'loss',
    '-l',
    type=str,
    default='mae',
    help='The loss to use for the training.'
)
@click.option(
    'refine_smaps',
    '-rfs',
    is_flag=True,
    help='Whether you want to use an smaps refiner.'
)
@click.option(
    'multicoil',
    '-mc',
    is_flag=True,
    help='Whether you want to use multicoil data.'
)
@click.option(
    'model',
    '-m',
    type=str,
    default='pdnet',
    help='The NC model to use.'
)
@click.option(
    'acq_type',
    '-t',
    type=str,
    default='radial',
    help='The trajectory to use.'
)
@click.option(
    'scale_factor',
    '-sf',
    type=float,
    default=1e6,
    help='The scale factor to use.'
)
@click.option(
    'three_d',
    '-3d',
    is_flag=True,
    help='Whether you want to use 3d data.'
)
@click.option(
    'dcomp',
    '-dc',
    is_flag=True,
    help='Whether you want to use density compensation.'
)
def train_ncnet_click(
        af,
        n_epochs,
        loss,
        refine_smaps,
        multicoil,
        model,
        acq_type,
        scale_factor,
        three_d,
        dcomp,
    ):
    train_ncnet_multinet(
        af=af,
        n_epochs=n_epochs,
        loss=loss,
        refine_smaps=refine_smaps,
        multicoil=multicoil,
        model=model,
        acq_type=acq_type,
        scale_factor=scale_factor,
        three_d=three_d,
        dcomp=dcomp,
    )

if __name__ == '__main__':
    train_ncnet_click()
