import os
import os.path as op
import time
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.callbacks import TQDMProgressBar
from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
from fastmri_recon.models.training.compile import default_model_compile
from fastmri_recon.models.utils.fastmri_format import tf_fastmri_format


n_volumes_train = 973
multicoil=True
acq_type='radial'
# acq_type='cartesian'
contrast=None
cuda_visible_devices='0123'
n_samples=1
n_epochs=1
n_iter=10
n_filters=32
n_primal=5
use_mixed_precision=False
non_linearity='relu'
loss='mae'
original_run_id=None
acq_kwargs = dict(
    af=10,
)
im_size = (640, 400)
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
    kwargs = acq_kwargs
else:
    dataset = singlecoil_dataset
    kwargs = acq_kwargs
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
train_set = dataset(
    train_path,
    im_size,
    acq_type=acq_type,
    compute_dcomp=True,
    contrast=contrast,
    inner_slices=4,
    rand=True,
    scale_factor=1e6,
    n_samples=n_samples,
    **kwargs
)
val_set = dataset(
    val_path,
    im_size,
    acq_type=acq_type,
    compute_dcomp=True,
    contrast=contrast,
    inner_slices=None,
    rand=True,
    scale_factor=1e6,
    **kwargs
)
run_params = {
    'n_primal': n_primal,
    'multicoil': multicoil,
    'activation': non_linearity,
    'n_iter': n_iter,
    'n_filters': n_filters,
    'im_size': im_size,
    'normalize_image': False,
    'dcomp': True,
}
if multicoil:
    updnet_type = 'ncpdnet_sense_'
else:
    updnet_type = 'ncpdnet_singlecoil_'
additional_info = f'{acq_type}'
if contrast is not None:
    additional_info += f'_{contrast}'
if n_samples is not None:
    additional_info += f'_{n_samples}'
if n_iter != 10:
    additional_info += f'_i{n_iter}'
if non_linearity != 'relu':
    additional_info += f'_{non_linearity}'
if loss != 'mae':
    additional_info += f'_{loss}'
run_id = f'{updnet_type}_{additional_info}_{int(time.time())}'
chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=True)
log_dir = op.join(f'{LOGS_DIR}logs', run_id)
tboard_cback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=False,
    write_images=False,
    profile_batch=20,
)
tqdm_cback = TQDMProgressBar()
model = NCPDNet(**run_params)
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
x, y = next(iter(train_set))
model.fit(
    x=x,
    y=y,
    epochs=1000,
    callbacks=[tqdm_cback, chkpt_cback],
    verbose=0,
)
model