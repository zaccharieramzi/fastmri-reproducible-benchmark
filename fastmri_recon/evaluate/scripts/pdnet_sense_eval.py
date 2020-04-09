from fastmri_recon.config import *


val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'

def evaluate_pdnet_sense(run_id='pdnet_sense_af4_1586266200', contrast=None, af=4, n_iter=10, n_samples=None):
    import tensorflow as tf

    from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
    from fastmri_recon.models.subclassed_models.pdnet import PDNet

    run_params = {
        'n_primal': 5,
        'n_dual': 1,
        'primal_only': True,
        'n_iter': n_iter,
        'multicoil': True,
        'n_filters': 32,
    }

    val_set = train_masked_kspace_dataset_from_indexable(
        val_path,
        AF=af,
        contrast=contrast,
        inner_slices=None,
        rand=False,
        scale_factor=1e6,
        parallel=False,
    )
    if n_samples is not None:
        val_set = val_set.take(n_samples)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = PDNet(**run_params)
        model([
            tf.zeros([1, 15, 640, 372, 1], dtype=tf.complex64),
            tf.zeros([1, 15, 640, 372], dtype=tf.complex64),
            tf.zeros([1, 15, 640, 372], dtype=tf.complex64),
        ])
        def tf_psnr(y_true, y_pred):
            perm_psnr = [3, 0, 1, 2]
            psnr = tf.image.psnr(
                tf.transpose(y_true, perm_psnr),
                tf.transpose(y_pred, perm_psnr),
                tf.reduce_max(y_true),
            )
            return psnr
        def tf_ssim(y_true, y_pred):
            perm_ssim = [3, 0, 1, 2]
            ssim = tf.image.ssim(
                tf.transpose(y_true, perm_ssim),
                tf.transpose(y_pred, perm_ssim),
                tf.reduce_max(y_true),
            )
            return ssim

        model.compile(loss=tf_psnr, metrics=[tf_ssim])
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-300.hdf5')
    eval_res = model.evaluate(val_set, verbose=1)
    return model.metrics_names, eval_res
