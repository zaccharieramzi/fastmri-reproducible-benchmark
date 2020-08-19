import os

import tensorflow as tf

from ..metrics.np_metrics import Metrics, METRIC_FUNCS
from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet


def evaluate_xpdnet(
        model_fun,
        model_kwargs,
        run_id,
        multicoil=True,
        brain=False,
        n_epochs=200,
        contrast=None,
        af=4,
        n_iter=10,
        res=True,
        n_scales=0,
        n_primal=5,
        refine_smaps=False,
        n_samples=None,
        cuda_visible_devices='0123',
        equidistant_fake=False,
    ):
    if multicoil:
        if brain:
            val_path = f'{FASTMRI_DATA_DIR}brain_multicoil_val/'
        else:
            val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
    else:
        val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    af = int(af)

    run_params = {
        'n_primal': n_primal,
        'multicoil': multicoil,
        'n_scales': n_scales,
        'n_iter': n_iter,
        'refine_smaps': refine_smaps,
        'res': res,
        'output_shape_spec': brain,
    }

    if multicoil:
        dataset = multicoil_dataset
        if equidistant_fake:
            mask_type = 'equidistant_fake'
        else:
            mask_type = 'equidistant'
        else:
            mask_type = 'random'
        kwargs = {
            'parallel': False,
            'output_shape_spec': brain,
            'mask_type': mask_type,
        }
    else:
        dataset = singlecoil_dataset
        kwargs = {}
    val_set = dataset(
        val_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=False,
        scale_factor=1e6,
        **kwargs,
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

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        if multicoil:
            kspace_size = [1, 15, 640, 372]
        else:
            kspace_size = [1, 640, 372]

        model = XPDNet(model_fun, model_kwargs, **run_params)
        inputs = [
            tf.zeros(kspace_size + [1], dtype=tf.complex64),
            tf.zeros(kspace_size, dtype=tf.complex64),
        ]
        if multicoil:
            inputs.append(tf.zeros(kspace_size, dtype=tf.complex64))
        model(inputs)
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    eval_res = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=n_volumes if n_samples is None else n_samples):
        y_pred = model.predict(x, batch_size=4)
        eval_res.push(y_true[..., 0], y_pred[..., 0])
    return METRIC_FUNCS, list(m.means().values())
