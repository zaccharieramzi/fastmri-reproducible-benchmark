import os

import click

from fastmri_recon.config import *


val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'

def evaluate_pdnet_sense(run_id='pdnet_sense_af4_1586266200', contrast=None, af=4, n_iter=10, n_samples=None, cuda_visible_devices='0123'):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

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
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-300.hdf5')
    eval_res = model.evaluate(val_set, verbose=1, steps=199 if n_samples is None else None)
    return model.metrics_names, eval_res

@click.command()
@click.option(
    'run_id',
    '-r',
    default='pdnet_sense_af4_1586266200',
    type=str,
    help='The run id of the trained network. Defaults to pdnet_sense_af4_1586266200.',
)
@click.option(
    'contrast',
    '-c',
    default=None,
    type=click.Choice(['CORPDFS_FBK', 'CORPD_FBK', None], case_sensitive=False),
    help='The contrast chosen for this evaluation. Defaults to CORPDFS_FBK.',
)
@click.option(
    'af',
    '-a',
    default='4',
    type=click.Choice(['4', '8']),
    help='The acceleration factor chosen for this fine tuning. Defaults to 4.',
)
@click.option(
    'n_iter',
    '-i',
    default=10,
    type=int,
    help='The number of epochs to train the model. Default to 300.',
)
@click.option(
    '-n',
    'n_samples',
    default=None,
    type=int,
    help='The number of samples to take from the dataset. Default to None (all samples taken).',
)
@click.option(
    'cuda_visible_devices',
    '-gpus',
    '--cuda-visible-devices',
    default='0123',
    type=str,
    help='The visible GPU devices. Defaults to 0123',
)
def evaluate_pdnet_sense_click(run_id, contrast, af, n_iter, cuda_visible_devices, n_samples):
    af = int(af)
    metrics_names, eval_res = evaluate_pdnet_sense(
        run_id=run_id,
        contrast=contrast,
        af=af,
        n_iter=n_iter,
        n_samples=n_samples,
        cuda_visible_devices=cuda_visible_devices,
    )
    print(metrics_names)
    print(eval_res)

if __name__ == '__main__':
    evaluate_pdnet_sense_click()
