import os

import click
import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.evaluate.metrics.np_metrics import Metrics, METRIC_FUNCS
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
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
        refine_big=False,
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
        'refine_big': refine_big,
    }

    if multicoil:
        dataset = multicoil_dataset
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

        model = XPDNet(model_fun, model_kwargs, **run_params)
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
    eval_res = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=n_volumes if n_samples is None else n_samples):
        y_pred = model.predict(x, batch_size=4)
        eval_res.push(y_true[..., 0], y_pred[..., 0])
    return METRIC_FUNCS, (list(eval_res.means().values()), list(eval_res.stddevs().values())))

@click.command()
@click.option(
    'model_name',
    '-m',
    type=str,
    default='MWCNN',
    help='The type of model you want to use for the XPDNet',
)
@click.option(
    'model_size',
    '-s',
    type=str,
    default='big',
    help='The size of the model you want to use for the XPDNet',
)
@click.option(
    'run_id',
    '-r',
    default=None,
    type=str,
    help='The run id of the trained network.',
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
    'refine_smaps',
    '-rfs',
    is_flag=True,
    help='Whether you want to use an smaps refiner.'
)
@click.option(
    'refine_big',
    '-rfsb',
    is_flag=True,
    help='Whether you want to use a big smaps refiner.'
)
@click.option(
    'brain',
    '-b',
    is_flag=True,
    help='Whether you want to consider brain data.'
)
@click.option(
    'equidistant_fake',
    '-eqf',
    is_flag=True,
    help='Whether you want to use fake equidistant masks for brain data.'
)
def evaluate_xpdnet_click(
        model_name,
        model_size,
        run_id,
        n_epochs,
        contrast,
        af,
        n_iter,
        cuda_visible_devices,
        n_samples,
        refine_smaps,
        refine_big,
        brain,
        equidistant_fake,
    ):
    n_primal = 5
    model_fun, kwargs, n_scales, res = [
         (model_fun, kwargs, n_scales, res)
         for m_name, m_size, model_fun, kwargs, _, n_scales, res
         in get_model_specs(n_primal=n_primal, force_res=False)
         if m_name == model_name and m_size == model_size
    ][0]

    metrics_names, eval_res = evaluate_xpdnet(
        model_fun=model_fun,
        model_kwargs=kwargs,
        multicoil=True,
        run_id=run_id,
        n_epochs=n_epochs,
        contrast=contrast,
        af=af,
        n_iter=n_iter,
        n_primal=n_primal,
        n_scales=n_scales,
        n_samples=n_samples,
        cuda_visible_devices=cuda_visible_devices,
        refine_smaps=refine_smaps or refine_big,
        refine_big=refine_big,
        brain=brain,
        equidistant_fake=equidistant_fake,
    )
    print(metrics_names)
    print(eval_res)

if __name__ == '__main__':
    evaluate_xpdnet_click()
