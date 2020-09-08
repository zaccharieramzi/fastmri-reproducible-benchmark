import os

from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import build_model_from_specs
from fastmri_recon.models.subclassed_models.multiscale_complex import MultiscaleComplex


def evaluate_xpdnet_dealiasing(
        model_fun,
        model_kwargs,
        run_id,
        n_scales=0,
        n_epochs=200,
        contrast='CORPD_FBK',
        af=4,
        n_samples=None,
        cuda_visible_devices='0123',
    ):

    val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    val_set = singlecoil_dataset(
        val_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=False,
        scale_factor=1e6,
    )
    if n_samples is not None:
        val_set = val_set.take(n_samples)
    else:
        val_set = val_set.take(199)

    model = MultiscaleComplex(
        model_fun=model_fun,
        model_kwargs=model_kwargs,
        res=False,
        n_scales=n_scales,
        fastmri_format=True,
    )
    model(next(iter(val_set))[0])
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    m = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=199 if n_samples is None else n_samples):
        y_pred = model.predict(x, batch_size=1)
        m.push(y_true[..., 0], y_pred[..., 0])
    return ['PSNR', 'SSIM'], list(m.means().values())
