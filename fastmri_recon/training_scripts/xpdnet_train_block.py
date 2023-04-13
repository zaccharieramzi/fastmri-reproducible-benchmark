from contextlib import ExitStack
from functools import partial
import math
import os
import os.path as op
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
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
from fastmri_recon.models.training.compile import default_model_compile
from fastmri_recon.training_scripts.custom_objects import CUSTOM_TF_OBJECTS
from fastmri_recon.training_scripts.model_saving_workaround import ModelCheckpointWorkAround


# this number means that 99.56% of all images will not be affected by
# cropping
# TODO: verify this number for brain
IM_SIZE = (640, 400)


def train_xpdnet_block(
        model_fun,
        model_kwargs,
        model_size=None,
        multicoil=True,
        brain=False,
        af=4,
        contrast=None,
        n_samples=None,
        batch_size=None,
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
        equidistant_fake=False,
        multi_gpu=False,
        mask_type=None,
        primal_only=True,
        n_dual=1,
        n_dual_filters=16,
        multiscale_kspace_learning=False,
        block_size=10,
        block_overlap=0,
        epochs_per_block_step=None,
    ):
    r"""Train an XPDNet network on the fastMRI dataset.

    The training is done with a learning rate of 1e-4, using the RAdam optimizer.
    The validation is performed every 5 epochs on 5 volumes.
    A scale factor of 1e6 is applied to the data.

    Arguments:
        model_fun (function): the function initializing the image correction
            network of the XPDNet.
        model_kwargs (dict): the set of arguments used to initialize the image
            correction network.
        model_size (str or None): a string describing the size of the network
            used. This is used in the run id. Defaults to None.
        multicoil (bool): whether the input data is multicoil. Defaults to False.
        brain (bool): whether to consider brain data instead of knee. Defaults
            to False.
        af (int): the acceleration factor for the retrospective undersampling
            of the data. Defaults to 4.
        contrast (str or None): the contrast used for this specific training.
            If None, all contrasts are considered. Defaults to None
        n_samples (int or None): the number of samples to consider for this
            training. If None, all samples are considered. Defaults to None.
        n_epochs (int): the number of epochs (i.e. one pass though all the
            volumes/samples) for this training. Defaults to 200.
        checkpoint_epoch (int): the number of epochs used to train the model
            during the first step of the full training. This is typically used
            when on a cluster the training duration exceeds the maximum job
            duration. Defaults to 0, which means that the training is done
            without checkpoints.
        save_state (bool): whether you should save the entire model state for
            this training, for example to retrain where left off. Defaults to
            False.
        n_iter (int): the number of iterations for the XPDNet.
        res (bool): whether the XPDNet image correction networks should be
            residual.
        n_scales (int): the number of scales used in the image correction
            network. Defaults to 0.
        n_primal (int): the size of the buffer in the image space. Defaults to
            5.
        use_mixed_precision (bool): whether to use the mixed precision API for
            training. Currently not working. Defaults to False.
        refine_smaps (bool): whether you want to refine the sensitivity maps
            with a neural network.
        loss (tf.keras.losses.Loss or str): the loss function used for the
            training. It should be understandable by the tf.keras loss API,
            or be 'compound_mssim', in which case the compound L1 MSSIM loss
            inspired by [P2020]. Defaults to 'mae'.
        original_run_id (str or None): run id of the same network trained before
            fine-tuning. If this is present, the training is considered
            fine-tuning for a network trained for 250 epochs. It will therefore
            apply a learning rate of 1e-7 and the epoch size will be divided in
            half. If None, the training is done normally, without loading
            weights. Defaults to None.
        fixed_masks (bool): whether fixed masks should be used for the
            retrospective undersampling. Defaults to False
        n_epochs_original (int): the number of epochs used to pre-train the
            model, only applicable if original_run_id is not None. Defaults to
            250.
        equidistant_fake (bool): whether to use fake equidistant masks from
            fastMRI. Defaults to False.
        multi_gpu (bool): whether to use multiple GPUs for the XPDNet training.
            Defaults to False.

    Returns:
        - str: the run id of the trained network.
    """
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
    mixed_precision.set_global_policy(policy)
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
        rand=True,
        scale_factor=1e6,
        n_samples=n_samples,
        fixed_masks=fixed_masks,
        batch_size=batch_size,
        target_image_size=IM_SIZE,
        **kwargs
    )
    val_set = dataset(
        val_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=True,
        scale_factor=1e6,
        **kwargs
    )

    run_params = {
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

    if multicoil:
        xpdnet_type = 'xpdnet_sense_'
        if brain:
            xpdnet_type += 'brain_'
    else:
        xpdnet_type = 'xpdnet_singlecoil_'
    additional_info = f'af{af}'
    if contrast is not None:
        additional_info += f'_{contrast}'
    if n_samples is not None:
        additional_info += f'_{n_samples}'
    if n_iter != 10:
        additional_info += f'_i{n_iter}'
    if loss != 'mae':
        additional_info += f'_{loss}'
    if refine_smaps:
        additional_info += '_rf_sm'
        if refine_big:
            additional_info += 'b'
    if fixed_masks:
        additional_info += '_fixed_masks'
    if block_overlap != 0:
        additional_info += f'_blkov{block_overlap}'

    submodel_info = model_fun.__name__
    if model_size is not None:
        submodel_info += model_size
    run_id = f'{xpdnet_type}_{additional_info}_bbb_{submodel_info}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}'
    chkpt_path += '.hdf5'

    log_dir = op.join(f'{LOGS_DIR}logs', run_id)
    tboard_cback = TensorBoard(
        profile_batch=0,
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
    )
    tqdm_cback = TQDMProgressBar()

    model = XPDNet(model_fun, model_kwargs, **run_params)
    n_steps = n_volumes

    if batch_size is not None:
        n_steps //= batch_size

    chkpt_cback = ModelCheckpointWorkAround(
        chkpt_path,
        save_freq=int(n_epochs*n_steps),
        save_weights_only=True,
    )
    print(run_id)
    stride = block_size - block_overlap
    assert stride > 0
    n_block_steps = int(math.ceil((n_iter - block_size) /  stride) + 1)
    ## epochs handling
    start_epoch = 0
    final_epoch = min(epochs_per_block_step, n_epochs)

    for i_step in range(n_block_steps):
        first_block_to_train = i_step * stride
        blocks = list(range(first_block_to_train, first_block_to_train + block_size))
        model.blocks_to_train = blocks
        default_model_compile(model, lr=lr, loss=loss)

        model.fit(
            train_set,
            steps_per_epoch=n_steps,
            initial_epoch=start_epoch,
            epochs=final_epoch,
            validation_data=val_set,
            validation_steps=5,
            validation_freq=5,
            verbose=0,
            callbacks=[tboard_cback, chkpt_cback, tqdm_cback],
        )
        n_epochs = n_epochs - (final_epoch - start_epoch)
        if n_epochs <= 0:
            break
        start_epoch = final_epoch
        final_epoch += min(epochs_per_block_step, n_epochs)
    return run_id
