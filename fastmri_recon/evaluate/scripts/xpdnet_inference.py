import os

import click
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import test_filenames
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import test_masked_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.data.datasets.fastmri_pyfunc import test_masked_kspace_dataset_from_indexable as singecoil_dataset
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
from fastmri_recon.evaluate.utils.write_results import write_result
from fastmri_recon.training_scripts.custom_objects import CUSTOM_TF_OBJECTS


def xpdnet_inference(
        model_fun,
        model_kwargs,
        run_id,
        multicoil=True,
        exp_id='xpdnet',
        brain=False,
        challenge=False,
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
        primal_only=True,
        n_dual=1,
        n_dual_filters=16,
        distributed=False,
        manual_saving=False,
    ):
    if brain:
        if challenge:
            test_path = f'{FASTMRI_DATA_DIR}brain_multicoil_challenge/'
        else:
            test_path = f'{FASTMRI_DATA_DIR}brain_multicoil_test/'
    else:
        if multicoil:
            test_path = f'{FASTMRI_DATA_DIR}multicoil_test_v2/'
        else:
            test_path = f'{FASTMRI_DATA_DIR}singlecoil_test/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    af = int(af)

    run_params = {
        'n_primal': n_primal,
        'multicoil': multicoil,
        'n_scales': n_scales,
        'n_iter': n_iter,
        'refine_smaps': refine_smaps,
        'refine_big': refine_big,
        'res': res,
        'output_shape_spec': brain,
        'primal_only': primal_only,
        'n_dual': n_dual,
        'n_dual_filters': n_dual_filters,
    }
    if multicoil:
        ds_fun = multicoil_dataset
        extra_kwargs = dict(output_shape_spec=brain)
    else:
        ds_fun = singecoil_dataset
        extra_kwargs = {}
    test_set = ds_fun(
        test_path,
        AF=af,
        contrast=contrast,
        scale_factor=1e6,
        n_samples=n_samples,
        **extra_kwargs
    )
    test_set_filenames = test_filenames(
        test_path,
        AF=af,
        contrast=contrast,
        n_samples=n_samples,
    )
    if multicoil:
        fake_kspace_size = [15, 640, 372]
    else:
        fake_kspace_size = [640, 372]
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        if distributed and not manual_saving:
            model = load_model(
                f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}',
                custom_objects=CUSTOM_TF_OBJECTS,
            )
        else:
            model = XPDNet(model_fun, model_kwargs, **run_params)
            fake_inputs = [
                tf.zeros([1, *fake_kspace_size, 1], dtype=tf.complex64),
                tf.zeros([1, *fake_kspace_size], dtype=tf.complex64),
            ]
            if multicoil:
                fake_inputs.append(tf.zeros([1, *fake_kspace_size], dtype=tf.complex64))
            if brain:
                fake_inputs.append(tf.constant([[320, 320]]))
            model(fake_inputs)
            model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    if n_samples is None:
        if not brain:
            if contrast:
                tqdm_total = n_volumes_test[af] // 2
            else:
                tqdm_total = n_volumes_test[af]
        else:
            if contrast:
                tqdm_total = brain_volumes_per_contrast['test'][af][contrast]
            else:
                tqdm_total = brain_n_volumes_test[af]
    else:
        tqdm_total = n_samples
    tqdm_desc = f'{exp_id}_{contrast}_{af}'

    for data_example, filename in tqdm(zip(test_set, test_set_filenames), total=tqdm_total, desc=tqdm_desc):
        res = model.predict(data_example, batch_size=16)
        write_result(
            exp_id,
            res,
            filename.numpy().decode('utf-8'),
            scale_factor=1e6,
            brain=brain,
            challenge=challenge,
            coiltype='multicoil' if multicoil else 'singlecoil',
        )


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
    'af',
    '-a',
    type=int,
    default=4,
    help='The acceleration factor.'
)
@click.option(
    'n_iter',
    '-i',
    default=10,
    type=int,
    help='The number of iterations of the unrolled model. Default to 10.',
)
@click.option(
    'brain',
    '-b',
    is_flag=True,
    help='Whether you want to consider brain data.'
)
@click.option(
    'challenge',
    '-ch',
    is_flag=True,
    help='Whether you want to consider challenge data (only for brain).'
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
    'n_epochs',
    '-e',
    type=int,
    default=10,
    help='The number of epochs used in the final training.'
)
@click.option(
    'run_id',
    '-r',
    type=str,
    default=None,
    help='The run id of the final training.'
)
@click.option(
    'exp_id',
    '-x',
    type=str,
    default='updnet',
    help='The experiment id.'
)
@click.option(
    'contrast',
    '-c',
    type=str,
    default=None,
    help='The contrast to use for the training.'
)
def xpdnet_inference_click(
        model_name,
        model_size,
        af,
        n_iter,
        brain,
        challenge,
        refine_smaps,
        refine_big,
        n_epochs,
        run_id,
        exp_id,
        contrast,
    ):
    n_primal = 5
    model_fun, kwargs, n_scales, res = [
         (model_fun, kwargs, n_scales, res)
         for m_name, m_size, model_fun, kwargs, _, n_scales, res
         in get_model_specs(n_primal=n_primal, force_res=False)
         if m_name == model_name and m_size == model_size
    ][0]

    xpdnet_inference(
        model_fun=model_fun,
        model_kwargs=kwargs,
        af=af,
        n_iter=n_iter,
        brain=brain,
        challenge=challenge,
        refine_smaps=refine_smaps or refine_big,
        refine_big=refine_big,
        n_epochs=n_epochs,
        run_id=run_id,
        exp_id=exp_id,
        contrast=contrast,
        res=res,
        n_scales=n_scales,
        n_primal=n_primal,
    )


if __name__ == '__main__':
    xpdnet_inference_click()
