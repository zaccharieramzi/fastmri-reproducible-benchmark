from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.models.subclassed_models.pdnet import PDNet

val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
epochs = 10
n_iter = 30

val_set = train_masked_kspace_dataset_from_indexable(
    val_path,
    AF=4,
    contrast=None,
    inner_slices=8,
    rand=True,
    scale_factor=1e6,
    parallel=False,
    n_samples=None,
)

model = PDNet(n_filters=4, n_primal=1, n_dual=1, primal_only=True, n_iter=1, multicoil=True, activation='linear')
model.compile(
    optimizer='adam',
    loss='mean_absolute_error',
)

model.fit(
    val_set,
    steps_per_epoch=n_iter,
    epochs=epochs,
    verbose=0,
)
