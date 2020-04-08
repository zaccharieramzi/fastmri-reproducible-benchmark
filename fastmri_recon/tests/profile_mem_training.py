import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.models.subclassed_models.pdnet import PDNet

val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
epochs = 50
n_iter = 30

af = 4
contrast = None
n_samples = n_iter

train_set = train_masked_kspace_dataset_from_indexable(
    val_path,
    AF=af,
    contrast=contrast,
    inner_slices=8,
    rand=True,
    scale_factor=1e6,
    n_samples=n_samples,
    parallel=False,
)
val_set = train_masked_kspace_dataset_from_indexable(
    val_path,
    AF=af,
    contrast=contrast,
    inner_slices=8,
    rand=True,
    scale_factor=1e6,
    parallel=False,
)

model = PDNet(n_filters=4, n_primal=1, n_dual=1, primal_only=True, n_iter=1, multicoil=True, activation='linear')
model.compile(
    optimizer='adam',
    loss='mean_absolute_error',
)

log_dir = op.join(f'{LOGS_DIR}logs', f'profiling_sess_{int(time.time())}')
tboard_cback = TensorBoard(
    profile_batch=0,
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=False,
    write_images=False,
)

model.fit(
    val_set,
    steps_per_epoch=n_iter,
    epochs=epochs,
    verbose=0,
    callbacks=[TQDMProgressBar(), tboard_cback],
)
