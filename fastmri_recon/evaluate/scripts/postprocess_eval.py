import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_postproc_tf_records import train_postproc_dataset_from_tfrecords
from fastmri_recon.evaluate.metrics.np_metrics import Metrics, METRIC_FUNCS
from fastmri_recon.models.subclassed_models.post_processing_3d import PostProcessVnet


def evaluate_vnet_postproc(
        original_run_id,
        run_id,
        brain=False,
        n_epochs=200,
        contrast=None,
        af=4,
        n_samples=None,
        base_n_filters=16,
        n_scales=4,
        non_linearity='prelu',
    ):
    if brain:
        val_path = f'{FASTMRI_DATA_DIR}brain_multicoil_val/'
    else:
        val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'

    af = int(af)
    run_params = dict(
        layers_n_channels=[base_n_filters*2**i for i in range(n_scales)],
        layers_n_non_lins=2,
        non_linearity=non_linearity,
        res=True,
    )
    model = PostProcessVnet(None, run_params)
    model(tf.zeros([2, 320, 320, 1]))
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    val_set = train_postproc_dataset_from_tfrecords(
        val_path,
        original_run_id,
        n_samples=n_samples,
    )
    if brain:
        n_volumes = brain_n_volumes_validation
        if contrast is not None:
            n_volumes = brain_volumes_per_contrast['validation'][contrast]
    else:
        n_volumes = n_volumes_val
        if contrast is not None:
            n_volumes //= 2
            n_volumes += 1
    if n_samples is not None:
        val_set = val_set.take(n_samples)
    else:
        val_set = val_set.take(n_volumes)


    eval_res = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=n_volumes if n_samples is None else n_samples):
        y_pred = model.predict_batched(x, batch_size=4)
        eval_res.push(y_true[..., 0], y_pred[..., 0])
    return METRIC_FUNCS, (list(eval_res.means().values()), list(eval_res.stddevs().values()))
