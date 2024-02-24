import math
import os
import os.path as op
import time

import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import mixed_precision
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.data.datasets.oasis_tf_records import train_nc_kspace_dataset_from_tfrecords as three_d_dataset
from fastmri_recon.data.datasets.multicoil.non_cartesian_tf_records import train_nc_kspace_dataset_from_tfrecords as multicoil_dataset
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
from fastmri_recon.models.training.compile import default_model_compile
from fastmri_recon.training_scripts.model_saving_workaround import ModelCheckpointWorkAround


n_volumes_train_fastmri = 973
n_volumes_train_oasis = 3273
# this number means that 99.56% of all images will not be affected by
# cropping
IM_SIZE = (640, 400)
VOLUME_SIZE = (176, 256, 256)

def train_ncnet_block(
        model,
        n_iter=10,
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
        block_size=10,
        block_overlap=0,
        epochs_per_block_step=None,
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
    if block_overlap != 0:
        additional_info += f'_blkov{block_overlap}'
    if checkpoint_epoch == 0:
        run_id = f'{run_id}_bbb_{additional_info}_{int(time.time())}'
    else:
        run_id = original_run_id
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
        save_freq=int(epochs_per_block_step*n_steps),
        save_optimizer=False,
        save_weights_only=True,
    )
    print(run_id)
    # if there are 4 blocks, with a block size of 2 and a block overlap of 1
    # we do the following block combinations:
    # 01, 12, 23 -> n block steps = 3
    # if there are 6 blocks with a block size 3 and a block overlap of 2:
    # 012, 123, 234, 345 -> n = 4
    # if there are 6 blocks with a block size 3 and a block overlap of 1:
    # 012, 234, 456 -> n = 3
    stride = block_size - block_overlap
    assert stride > 0
    n_block_steps = int(math.ceil((n_iter - block_size) /  stride) + 1)
    ## epochs handling
    restart_at_block_step = checkpoint_epoch // epochs_per_block_step
    start_epoch = checkpoint_epoch
    final_epoch = checkpoint_epoch + min(epochs_per_block_step, n_epochs)
    for i_step in range(n_block_steps):
        if i_step < restart_at_block_step:
            continue
        first_block_to_train = i_step * stride
        blocks = list(range(first_block_to_train, first_block_to_train + block_size))
        model.blocks_to_train = blocks
        default_model_compile(model, lr=lr, loss=loss)
        # first run of the model to avoid the saving error
        # ValueError: as_list() is not defined on an unknown TensorShape.
        # it can also allow loading of weights
        model(next(iter(train_set))[0])
        if not checkpoint_epoch == 0 and i_step == restart_at_block_step:
            model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{original_run_id}-{checkpoint_epoch:02d}.hdf5')
            if not checkpoint_epoch % epochs_per_block_step == 0:
                grad_vars = model.trainable_weights
                zero_grads = [tf.zeros_like(w) for w in grad_vars]
                model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
                with open(f'{CHECKPOINTS_DIR}checkpoints/{original_run_id}-optimizer.pkl', 'rb') as f:
                    weight_values = pickle.load(f)
                model.optimizer.set_weights(weight_values)
        model.fit(
            train_set,
            steps_per_epoch=n_steps,
            initial_epoch=start_epoch,
            epochs=final_epoch,
            validation_data=val_set,
            validation_steps=5,
            verbose=0,
            callbacks=[tboard_cback, chkpt_cback, tqdm_cback],
        )
        n_epochs = n_epochs - (final_epoch - start_epoch)
        if n_epochs <= 0:
            break
        start_epoch = final_epoch
        final_epoch += min(epochs_per_block_step, n_epochs)
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

    return train_ncnet_block(
        model,
        n_iter=n_iter,
        run_id=run_id,
        multicoil=multicoil,
        dcomp=dcomp,
        three_d=three_d,
        **train_kwargs,
    )
