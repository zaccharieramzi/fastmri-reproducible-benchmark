from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable

val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
val_set = train_masked_kspace_dataset_from_indexable(
    val_path,
    AF=4,
    contrast=None,
    inner_slices=None,
    rand=False,
    scale_factor=1e6,
    parallel=False,
)

next(iter(val_set))
