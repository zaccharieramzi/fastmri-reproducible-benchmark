import os

import click
import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from fastmri_recon.models.subclassed_models.updnet import UPDNet


def evaluate_updnet(
        multicoil=True,
        brain=False,
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
        verbose=False,
        equidistant_fake=False,
        mask_type=None,
    ):
    if verbose:
        print(f'Evaluating {run_id}')
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
        'n_primal': 5,
        'n_dual': 1,
        'primal_only': True,
        'multicoil': multicoil,
        'n_layers': n_layers,
        'layers_n_channels': [base_n_filter * 2**i for i in range(n_layers)],
        'non_linearity': non_linearity,
        'n_iter': n_iter,
        'channel_attention_kwargs': channel_attention_kwargs,
        'refine_smaps': refine_smaps,
        'output_shape_spec': brain,
    }

    if multicoil:
        dataset = multicoil_dataset
        if mask_type is None:
            if brain:
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
        model = UPDNet(**run_params)
        inputs = [
            tf.zeros(kspace_size + [1], dtype=tf.complex64),
            tf.zeros(kspace_size, dtype=tf.complex64),
        ]
        if multicoil:
            inputs.append(tf.zeros(kspace_size, dtype=tf.complex64))
        if brain:
            inputs.append(tf.constant([[320, 320]]))
        model(inputs)
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    m = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=n_volumes if n_samples is None else n_samples):
        y_pred = model.predict(x, batch_size=4)
        m.push(y_true[..., 0], y_pred[..., 0])
    return METRIC_FUNCS, list(m.means().values())

@click.command()
@click.option(
    'run_id',
    '-r',
    default=None,
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
    type=str,
    help='The contrast chosen for this evaluation. Defaults to None.',
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
@click.option(
    'refine_smaps',
    '-rfs',
    is_flag=True,
    help='Whether you want to use an smaps refiner.'
)
@click.option(
    'brain',
    '-b',
    is_flag=True,
    help='Whether you want to consider brain data.'
)
@click.option(
    'verbose',
    '-v',
    is_flag=True,
    help='Whether to print some logging info.',
)
@click.option(
    'equidistant_fake',
    '-eqf',
    is_flag=True,
    help='Whether you want to use fake equidistant masks for brain data.'
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
        refine_smaps,
        brain,
        verbose,
        equidistant_fake,
    ):
    if channel_attention == 'dense':
        channel_attention_kwargs = {'dense': True}
    elif channel_attention == 'conv':
        channel_attention_kwargs = {'dense': False}
    else:
        channel_attention_kwargs = None
    metrics_names, eval_res = evaluate_updnet(
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
        refine_smaps=refine_smaps,
        brain=brain,
        verbose=verbose,
        equidistant_fake=equidistant_fake,
    )
    print(metrics_names)
    print(eval_res)

if __name__ == '__main__':
    evaluate_updnet_sense_click()
