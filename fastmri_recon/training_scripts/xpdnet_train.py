import os
import os.path as op
import time

import click
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import load_model
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
from fastmri_recon.models.training.compile import default_model_compile
from .custom_objects import CUSTOM_TF_OBJECTS
from .model_saving_workaround import ModelCheckpointWorkAround


def train_xpdnet(
        model_fun,
        model_kwargs,
        model_size=None,
        multicoil=True,
        brain=False,
        af=4,
        contrast=None,
        cuda_visible_devices='0123',
        n_samples=None,
        n_epochs=200,
        checkpoint_epoch=0,
        save_state=False,
        n_iter=10,
        res=True,
        n_scales=0,
        n_primal=5,
        use_mixed_precision=False,
        refine_smaps=False,
        loss='mae',
        original_run_id=None,
        fixed_masks=False,
        n_epochs_original=250,
        equidistant_fake=False,
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
        cuda_visible_devices (str): the GPUs to consider visible. Defaults to
            '0123'.
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


    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
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
    if fixed_masks:
        additional_info += '_fixed_masks'

    submodel_info = model_fun.__name__
    if model_size is not None:
        submodel_info += model_size
    if checkpoint_epoch == 0:
        run_id = f'{xpdnet_type}_{additional_info}_{submodel_info}_{int(time.time())}'
    else:
        run_id = original_run_id
    final_epoch = checkpoint_epoch + n_epochs
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}'
    if not save_state:
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

    if checkpoint_epoch == 0:
        model = XPDNet(model_fun, model_kwargs, **run_params)
        if original_run_id is not None:
            lr = 1e-7
            n_steps = brain_volumes_per_contrast['train'].get(contrast, n_volumes)//2
        else:
            lr = 1e-4
            n_steps = n_volumes
        default_model_compile(model, lr=lr, loss=loss)
    else:
        model = load_model(
            f'{CHECKPOINTS_DIR}checkpoints/{original_run_id}-{checkpoint_epoch:02d}',
            custom_objects=CUSTOM_TF_OBJECTS,
        )
    chkpt_cback = ModelCheckpointWorkAround(
        chkpt_path,
        save_freq=n_epochs*n_steps,
        save_weights_only=not save_state,
    )
    print(run_id)
    if original_run_id is not None:
        if os.environ.get('FASTMRI_DEBUG'):
            n_epochs_original = 1
        model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{original_run_id}-{n_epochs_original:02d}.hdf5')

    model.fit(
        train_set,
        steps_per_epoch=n_steps,
        initial_epoch=checkpoint_epoch,
        epochs=final_epoch,
        validation_data=val_set,
        validation_steps=5,
        validation_freq=5,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback, tqdm_cback],
    )
    return run_id


@click.command()
@click.option(
    'model_name',
    '-m',
    type=str,
    default='MWCNN',
    help='The type of model you want to use for the XPDNet',
)
@click.option(
    'model_size',
    '-s',
    type=str,
    default='big',
    help='The size of the model you want to use for the XPDNet',
)
@click.option(
    'af',
    '-a',
    type=int,
    default=4,
    help='The acceleration factor.'
)
@click.option(
    'brain',
    '-b',
    is_flag=True,
    help='Whether you want to consider brain data.'
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
    'n_epochs',
    '-e',
    type=int,
    default=200,
    help='The number of epochs used for training.'
)
@click.option(
    'checkpoint_epoch',
    '-ec',
    type=int,
    default=0,
    help='The number of epochs used in the first step of training.'
)
@click.option(
    'n_epochs_original',
    '--n-epochs-orig',
    type=int,
    default=200,
    help='The number of epochs used in the original unspecific training.'
)
@click.option(
    'save_state',
    '-ss',
    is_flag=True,
    help='Whether you want to save the model state.'
)
@click.option(
    'original_run_id',
    '--orig-id',
    type=str,
    default=None,
    help='The run id of the original unspecific training.'
)
@click.option(
    'contrast',
    '-c',
    type=str,
    default=None,
    help='The contrast to use for the training.'
)
@click.option(
    'equidistant_fake',
    '-eqf',
    is_flag=True,
    help='Whether you want to use fake equidistant masks for brain data.'
)
def train_xpdnet_click(
        model_name,
        model_size,
        af,
        brain,
        loss,
        refine_smaps,
        n_epochs,
        checkpoint_epoch,
        n_epochs_original,
        save_state,
        original_run_id,
        contrast,
        equidistant_fake,
    ):
    n_primal = 5
    model_fun, kwargs, n_scales, res = [
         (model_fun, kwargs, n_scales, res)
         for m_name, m_size, model_fun, kwargs, _, n_scales, res in get_model_specs(n_primal=n_primal, force_res=False)
         if m_name == model_name and m_size == model_size
    ][0]

    train_xpdnet(
        model_fun=model_fun,
        model_kwargs=kwargs,
        model_size=model_size,
        multicoil=True,
        af=af,
        brain=brain,
        res=res,
        loss=loss,
        refine_smaps=refine_smaps,
        n_scales=n_scales,
        n_primal=n_primal,
        n_epochs=n_epochs,
        checkpoint_epoch=checkpoint_epoch,
        n_epochs_original=n_epochs_original,
        save_state=save_state,
        original_run_id=original_run_id,
        contrast=contrast,
        equidistant_fake=equidistant_fake,
    )


if __name__ == '__main__':
    train_xpdnet_click()
