import os
import os.path as op
import time

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.models.subclassed_models.post_processing_3d import PostProcessVnet
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
from fastmri_recon.models.training.compile import default_model_compile
from fastmri_recon.training_scripts.model_saving_workaround import ModelCheckpointWorkAround


# this number means that 99.56% of all images will not be affected by
# cropping
# TODO: verify this number for brain
IM_SIZE = (640, 400)


def train_vnet_postproc(
        orig_model_fun,
        orig_model_kwargs,
        original_run_id,
        multicoil=True,
        brain=False,
        af=4,
        contrast=None,
        n_samples=None,
        n_epochs=200,
        n_iter=10,
        res=True,
        n_scales=0,
        n_primal=5,
        use_mixed_precision=False,
        refine_smaps=False,
        refine_big=False,
        loss='mae',
        lr=1e-4,
        fixed_masks=False,
        n_epochs_original=250,
        equidistant_fake=False,
        multi_gpu=False,
        mask_type=None,
        primal_only=True,
        n_dual=1,
        n_dual_filters=16,
        multiscale_kspace_learning=False,
    ):
    if brain:
        n_volumes = brain_n_volumes_train
    else:
        n_volumes = n_volumes_train
    # paths
    if multicoil:
        if brain:
            train_path = f'{FASTMRI_DATA_DIR}brain_multicoil_train/'
            val_path = f'{FASTMRI_DATA_DIR}brain_multicoil_val/'
        else:
            train_path = f'{FASTMRI_DATA_DIR}multicoil_train/'
            val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
    else:
        train_path = f'{FASTMRI_DATA_DIR}singlecoil_train/singlecoil_train/'
        val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'


    af = int(af)

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
        if mask_type is None:
            if brain:
                if equidistant_fake:
                    mask_type = 'equidistant_fake'
                else:
                    mask_type = 'equidistant'
            else:
                mask_type = 'random'
        kwargs = {
            'parallel': False,
            'output_shape_spec': brain,
            'mask_type': mask_type,
        }
    else:
        dataset = singlecoil_dataset
        kwargs = {}
    train_set = dataset(
        train_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=False,
        scale_factor=1e6,
        n_samples=n_samples,
        fixed_masks=fixed_masks,
        target_image_size=IM_SIZE,
        **kwargs
    )
    val_set = dataset(
        val_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=False,
        scale_factor=1e6,
        **kwargs
    )

    orig_run_params = {
        'n_primal': n_primal,
        'multicoil': multicoil,
        'n_scales': n_scales,
        'n_iter': n_iter,
        'refine_smaps': refine_smaps,
        'res': res,
        'output_shape_spec': brain,
        'multi_gpu': multi_gpu,
        'refine_big': refine_big,
        'primal_only': primal_only,
        'n_dual': n_dual,
        'n_dual_filters': n_dual_filters,
        'multiscale_kspace_learning': multiscale_kspace_learning,
    }
    recon_model = XPDNet(orig_model_fun, orig_model_kwargs, **orig_run_params)
    n_steps = n_volumes
    default_model_compile(recon_model, lr=lr, loss=loss)
    if os.environ.get('FASTMRI_DEBUG'):
        n_epochs_original = 1
    if multicoil:
        kspace_size = [1, 15, 640, 372]
    else:
        kspace_size = [1, 640, 372]
    inputs = [
        tf.zeros(kspace_size + [1], dtype=tf.complex64),
        tf.zeros(kspace_size, dtype=tf.complex64),
    ]
    if multicoil:
        inputs.append(tf.zeros(kspace_size, dtype=tf.complex64))
    if brain:
        inputs.append(tf.constant([[320, 320]]))
    recon_model(inputs)
    recon_model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{original_run_id}-{n_epochs_original:02d}.hdf5')

    run_params = dict(
        layers_n_channels=[16, 32, 64, 128],
        layers_n_non_lins=2,
        non_linearity='prelu',
    )
    model = PostProcessVnet(recon_model, run_params)

    vnet_type = 'vnet_postproc_'
    if brain:
        vnet_type += 'brain_'
    additional_info = f'af{af}'
    if contrast is not None:
        additional_info += f'_{contrast}'
    if n_samples is not None:
        additional_info += f'_{n_samples}'
    if loss != 'mae':
        additional_info += f'_{loss}'
    run_id = f'{vnet_type}_{additional_info}_{int(time.time())}'

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

    chkpt_cback = ModelCheckpointWorkAround(
        chkpt_path,
        save_freq=n_epochs*n_steps,
        save_weights_only=True,
    )
    print(run_id)


    model.fit(
        train_set,
        steps_per_epoch=n_steps,
        epochs=n_epochs,
        validation_data=val_set,
        validation_steps=10,
        validation_freq=5,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback, tqdm_cback],
    )
    return run_id
