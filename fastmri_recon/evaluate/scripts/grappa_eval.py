from tqdm.notebook import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from fastmri_recon.evaluate.reconstruction.grappa_reconstruction import reco_grappa


def eval_grappa(af=4, contrast=None, n_samples=10):
    val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
    val_set = train_masked_kspace_dataset_from_indexable(
        val_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=False,
        scale_factor=1e6,
        n_samples=None,
        fixed_masks=False,
    )
    m = Metrics(METRIC_FUNCS)
    for (kspace, _), gt_image in tqdm(val_set.take(n_samples).as_numpy_iterator(), total=n_samples):
        reco = reco_grappa(kspace[..., 0], af=af)
        m.push(gt_image[..., 0], reco)
    return m
