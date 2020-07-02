import os

import tensorflow as tf

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as singlecoil_dataset
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
    sub_model = build_model_from_specs(model_fun, model_kwargs, 2)
    model = MultiscaleComplex(
        sub_model,
        res=False,
        n_scales=n_scales,
        fastmri_format=True,
    )
    model.compile(loss=tf_psnr, metrics=[tf_ssim])
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    eval_res = model.evaluate(val_set, verbose=1, steps=199 if n_samples is None else None)
    return model.metrics_names, eval_res
