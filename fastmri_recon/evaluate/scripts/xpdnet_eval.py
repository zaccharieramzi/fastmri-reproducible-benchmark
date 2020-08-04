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
        if brain:
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
    def tf_psnr(y_true, y_pred):
        perm_psnr = [3, 1, 2, 0]
        psnr = tf.image.psnr(
            tf.transpose(y_true, perm_psnr),
            tf.transpose(y_pred, perm_psnr),
            tf.reduce_max(y_true),
        )
        return psnr
    def tf_ssim(y_true, y_pred):
        perm_ssim = [0, 1, 2, 3]
        ssim = tf.image.ssim(
            tf.transpose(y_true, perm_ssim),
            tf.transpose(y_pred, perm_ssim),
            tf.reduce_max(y_true),
        )
        return ssim

    model.compile(loss=tf_psnr, metrics=[tf_ssim])
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    try:
        eval_res = model.evaluate(val_set, verbose=1, steps=n_volumes if n_samples is None else None)
    except tf.errors.ResourceExhaustedError:
        eval_res = Metrics(METRIC_FUNCS)
        if n_samples is None:
            val_set = val_set.take(n_volumes)
        for data in val_set:
            y_true = data[1].numpy()
            y_pred = model.predict(data[0], batch_size=1)
            eval_res.push(y_true[..., 0], y_pred[..., 0])
        eval_res = [eval_res.means()['PSNR'], eval_res.means()['SSIM']]
    return model.metrics_names, eval_res
