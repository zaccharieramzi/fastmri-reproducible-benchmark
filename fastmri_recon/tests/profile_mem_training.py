from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable

val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
epochs = 5
n_iter = 10

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

for e in range(epochs):
    training_iter = iter(val_set)
    for i in range(n_iter):
        res = next(training_iter)
