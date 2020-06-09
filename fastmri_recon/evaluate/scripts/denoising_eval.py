import os

import tensorflow as tf

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc_denoising import train_noisy_dataset_from_indexable


def evaluate_xpdnet_denoising(
        model,
        run_id,
        n_epochs=200,
        contrast='CORPD_FBK',
        noise_std=30,
        n_samples=None,
        cuda_visible_devices='0123',
    ):

    val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    val_set = train_noisy_dataset_from_indexable(
        val_path,
        noise_std=noise_std,
        contrast=contrast,
        inner_slices=None,
        rand=True,
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

    model.compile(loss=tf_psnr, metrics=[tf_ssim])
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    eval_res = model.evaluate(val_set, verbose=1, steps=199 if n_samples is None else None)
    return model.metrics_names, eval_res
