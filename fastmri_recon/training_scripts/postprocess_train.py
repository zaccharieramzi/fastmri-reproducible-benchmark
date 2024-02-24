import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import as mixed_precision
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_postproc_tf_records import train_postproc_dataset_from_tfrecords
from fastmri_recon.models.subclassed_models.post_processing_3d import PostProcessVnet
from fastmri_recon.models.training.compile import default_model_compile
from fastmri_recon.training_scripts.model_saving_workaround import ModelCheckpointWorkAround


# this number means that 99.56% of all images will not be affected by
# cropping
# TODO: verify this number for brain
IM_SIZE = (640, 400)


def train_vnet_postproc(
        original_run_id,
        af=4,
        brain=False,
        n_samples=None,
        n_epochs=200,
        use_mixed_precision=False,
        loss='mae',
        lr=1e-4,
        base_n_filters=16,
        n_scales=4,
        non_linearity='prelu',
    ):
    if brain:
        n_volumes = brain_n_volumes_train
    else:
        n_volumes = n_volumes_train
    # paths
    if brain:
        train_path = f'{FASTMRI_DATA_DIR}brain_multicoil_train/'
        val_path = f'{FASTMRI_DATA_DIR}brain_multicoil_val/'
    else:
        train_path = f'{FASTMRI_DATA_DIR}multicoil_train/'
        val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'

    # trying mixed precision
    if use_mixed_precision:
        policy_type = 'mixed_float16'
    else:
        policy_type = 'float32'
    policy = mixed_precision.Policy(policy_type)
    mixed_precision.set_global_policy(policy)
    # generators
    train_set = train_postproc_dataset_from_tfrecords(
        train_path,
        original_run_id,
        n_samples=n_samples,
    )
    val_set = train_postproc_dataset_from_tfrecords(
        val_path,
        original_run_id,
        n_samples=n_samples,
    )
    run_params = dict(
        layers_n_channels=[base_n_filters*2**i for i in range(n_scales)],
        layers_n_non_lins=2,
        non_linearity=non_linearity,
        res=True,
    )
    model = PostProcessVnet(None, run_params)
    default_model_compile(model, lr=lr, loss=loss)

    vnet_type = 'vnet_postproc_'
    if brain:
        vnet_type += 'brain_'
    additional_info = f'af{af}'
    if n_samples is not None:
        additional_info += f'_{n_samples}'
    if loss != 'mae':
        additional_info += f'_{loss}'
    if base_n_filters != 16:
        additional_info += f'_bf{base_n_filters}'
    if n_scales != 4:
        additional_info += f'_sc{n_scales}'
    if non_linearity != 'prelu':
        additional_info += f'_{non_linearity}'
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
        save_freq=n_epochs*n_volumes,
        save_weights_only=True,
    )
    print(run_id)


    model.fit(
        train_set,
        steps_per_epoch=n_volumes,
        epochs=n_epochs,
        validation_data=val_set,
        validation_steps=10,
        validation_freq=5,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback, tqdm_cback],
    )
    return run_id
