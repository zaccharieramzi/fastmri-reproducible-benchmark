import os

import click
import tensorflow as tf

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.models.subclassed_models.updnet import UPDNet


def evaluate_updnet_sense(
        run_id='updnet_sense_af4_1588609141',
        n_epochs=200,
        contrast=None,
        af=4,
        n_iter=10,
        n_layers=3,
        base_n_filter=16,
        non_linearity='relu',
        channel_attention_kwargs=None,
        refine_smaps=False,
        n_samples=None,
        cuda_visible_devices='0123',
    ):
    val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    af = int(af)

    run_params = {
        'n_primal': 5,
        'n_dual': 1,
        'primal_only': True,
        'multicoil': True,
        'n_layers': n_layers,
        'layers_n_channels': [base_n_filter * 2**i for i in range(n_layers)],
        'non_linearity': non_linearity,
        'n_iter': n_iter,
        'channel_attention_kwargs': channel_attention_kwargs,
        'refine_smaps': refine_smaps,
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
        model = UPDNet(**run_params)
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
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    eval_res = model.evaluate(val_set, verbose=1, steps=199 if n_samples is None else None)
    return model.metrics_names, eval_res

@click.command()
@click.option(
    'run_id',
    '-r',
    default='updnet_sense_af4_1588609141',
    type=str,
    help='The run id of the trained network. Defaults to updnet_sense_af4_1588609141.',
)
@click.option(
    'n_epochs',
    '-e',
    default=200,
    type=int,
    help='The number of epochs for which the model was trained or fine-tuned. Defaults to 200.',
)
@click.option(
    'contrast',
    '-c',
    default=None,
    type=click.Choice(['CORPDFS_FBK', 'CORPD_FBK',], case_sensitive=False),
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
@click.option(
    'non_linearity',
    '-nl',
    default='relu',
    type=str,
    help='The non linearity to use in the model. Default to relu.',
)
@click.option(
    'n_layers',
    '-la',
    default=3,
    type=int,
    help='The number of layers in the u-net. Default to 3.',
)
@click.option(
    'base_n_filter',
    '-nf',
    default=16,
    type=int,
    help='The number of base filters in the u-net (x2 each layer). Default to 16.',
)
@click.option(
    'channel_attention',
    '-ca',
    default=None,
    type=click.Choice([None, 'dense', 'conv']),
    help='The type of channel attention to use. Default to None.',
)
def evaluate_updnet_sense_click(
        run_id,
        n_epochs,
        contrast,
        af,
        n_iter,
        cuda_visible_devices,
        n_samples,
        non_linearity,
        n_layers,
        base_n_filter,
        channel_attention,
    ):
    if channel_attention == 'dense':
        channel_attention_kwargs = {'dense': True}
    elif channel_attention == 'conv':
        channel_attention_kwargs = {'dense': False}
    else:
        channel_attention_kwargs = None
    metrics_names, eval_res = evaluate_updnet_sense(
        run_id=run_id,
        n_epochs=n_epochs,
        contrast=contrast,
        af=af,
        n_iter=n_iter,
        non_linearity=non_linearity,
        n_layers=n_layers,
        base_n_filter=base_n_filter,
        channel_attention_kwargs=channel_attention_kwargs,
        n_samples=n_samples,
        cuda_visible_devices=cuda_visible_devices,
    )
    print(metrics_names)
    print(eval_res)

if __name__ == '__main__':
    evaluate_updnet_sense_click()
