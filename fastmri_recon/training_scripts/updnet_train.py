import os
import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.models.subclassed_models.updnet import UPDNet
from fastmri_recon.models.training.compile import default_model_compile


def train_updnet(
        multicoil=True,
        brain=False,
        af=4,
        contrast=None,
        cuda_visible_devices='0123',
        n_samples=None,
        n_epochs=200,
        n_iter=10,
        use_mixed_precision=False,
        n_layers=3,
        base_n_filter=16,
        non_linearity='relu',
        channel_attention_kwargs=None,
        refine_smaps=False,
        loss='mae',
        original_run_id=None,
        fixed_masks=False,
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
        kwargs = {'parallel': False}
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
        output_shape_spec=brain,
        **kwargs
    )
    val_set = dataset(
        val_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=True,
        scale_factor=1e6,
        output_shape_spec=brain,
        **kwargs
    )

    run_params = {
        'n_primal': 5,
        'n_dual': 1,
        'primal_only': True,
        'multicoil': multicoil,
        'n_layers': n_layers,
        'layers_n_channels': [base_n_filter * 2**i for i in range(n_layers)],
        'non_linearity': non_linearity,
        'n_iter': n_iter,
        'channel_attention_kwargs': channel_attention_kwargs,
        'refine_smaps': refine_smaps,
        'output_shape_spec': brain,
    }

    if multicoil:
        updnet_type = 'updnet_sense_'
        if brain:
            updnet_type += 'brain_'
    else:
        updnet_type = 'updnet_singlecoil_'
    additional_info = f'af{af}'
    if contrast is not None:
        additional_info += f'_{contrast}'
    if n_samples is not None:
        additional_info += f'_{n_samples}'
    if n_iter != 10:
        additional_info += f'_i{n_iter}'
    if non_linearity != 'relu':
        additional_info += f'_{non_linearity}'
    if n_layers != 3:
        additional_info += f'_l{n_layers}'
    if base_n_filter != 16:
        additional_info += f'_bf{base_n_filter}'
    if loss != 'mae':
        additional_info += f'_{loss}'
    if channel_attention_kwargs:
        additional_info += '_ca'
    if refine_smaps:
        additional_info += '_rf_sm'
    if fixed_masks:
        additional_info += '_fixed_masks'

    run_id = f'{updnet_type}_{additional_info}_{int(time.time())}'
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

    model = UPDNet(**run_params)
    if original_run_id is not None:
        lr = 1e-7
        n_steps = n_volumes//2
    else:
        lr = 1e-4
        n_steps = n_volumes
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
