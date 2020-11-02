import os

import tensorflow as tf
from tf_fastmri_data.datasets.noisy import NoisyFastMRIDatasetBuilder
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.evaluate.metrics.np_metrics import Metrics, METRIC_FUNCS
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import build_model_from_specs


tf.random.set_seed(0)

def evaluate_xpdnet_denoising(
        model,
        run_id,
        n_epochs=200,
        contrast='CORPD_FBK',
        noise_std=30,
        n_samples=100,
        cuda_visible_devices='0123',
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    val_set = NoisyFastMRIDatasetBuilder(
        dataset='val',
        noise_power_spec=noise_std,
        noise_input=False,
        contrast=contrast,
        slice_random=True,
        scale_factor=1e4,
        force_determinism=True,
    ).preprocessed_ds
    val_set = val_set.take(n_samples)

    if isinstance(model, tuple):
        model = build_model_from_specs(*model)
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    eval_res = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=n_samples):
        y_pred = model.predict(x)
        eval_res.push(y_true[..., 0], y_pred[..., 0])
    return METRIC_FUNCS, (list(eval_res.means().values()), list(eval_res.stddevs().values()))
