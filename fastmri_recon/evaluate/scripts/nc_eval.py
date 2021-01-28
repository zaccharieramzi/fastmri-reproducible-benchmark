import os
from pathlib import Path

import click
import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.data.datasets.oasis_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as three_d_dataset
from fastmri_recon.data.datasets.multicoil.non_cartesian_tf_records import train_nc_kspace_dataset_from_tfrecords as multicoil_dataset
from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from fastmri_recon.evaluate.reconstruction.non_cartesian_dcomp_reconstruction import NCDcompReconstructor
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
from fastmri_recon.models.subclassed_models.unet import UnetComplex
from fastmri_recon.models.subclassed_models.vnet import VnetComplex


# this number means that 99.56% of all images will not be affected by
# cropping
IM_SIZE = (640, 400)
VOLUME_SIZE = (256, 256, 256)

def _extract_first_elem_of_batch(inputs):
    if isinstance(inputs, (list, tuple)):
        return [_extract_first_elem_of_batch(i) for i in inputs]
    else:
        return inputs[0:1]

def evaluate_nc(
        model,
        multicoil=False,
        three_d=False,
        run_id=None,
        n_epochs=200,
        contrast=None,
        acq_type='radial',
        n_samples=None,
        cuda_visible_devices='0123',
        dcomp=False,
        **acq_kwargs,
    ):
    if multicoil:
        val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
    elif three_d:
        val_path = f'{OASIS_DATA_DIR}/val/'
    else:
        val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    if multicoil:
        dataset = multicoil_dataset
        image_size = IM_SIZE
    elif three_d:
        dataset = three_d_dataset
        image_size = VOLUME_SIZE
    else:
        dataset = singlecoil_dataset
        image_size = IM_SIZE
    if not three_d:
        add_kwargs = {
            'contrast': contrast,
            'rand': False,
            'inner_slices': None,
        }
    else:
        add_kwargs = {}
    add_kwargs.update(**acq_kwargs)
    val_set = dataset(
        val_path,
        image_size,
        acq_type=acq_type,
        compute_dcomp=dcomp,
        scale_factor=1e6,
        **add_kwargs
    )
    if n_samples is not None:
        val_set = val_set.take(n_samples)
    else:
        val_set = val_set.take(199)

    example_input = next(iter(val_set))[0]
    inputs = _extract_first_elem_of_batch(example_input)
    model(inputs)
    if run_id is not None:
        model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    if three_d:
        m = Metrics({'PSNR': METRIC_FUNCS['PSNR']})
    else:
        m = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=199 if n_samples is None else n_samples):
        y_pred = model.predict(x, batch_size=1)
        m.push(y_true[..., 0], y_pred[..., 0])
    print(METRIC_FUNCS.keys())
    print(list(m.means().values()))
    return METRIC_FUNCS, list(m.means().values())

def evaluate_ncpdnet(
        multicoil=False,
        three_d=False,
        dcomp=False,
        normalize_image=False,
        n_iter=10,
        n_filters=32,
        n_primal=5,
        non_linearity='relu',
        refine_smaps=True,
        **eval_kwargs
    ):
    if three_d:
        image_size = VOLUME_SIZE
    else:
        image_size = IM_SIZE
    run_params = {
        'n_primal': n_primal,
        'multicoil': multicoil,
        'three_d': three_d,
        'activation': non_linearity,
        'n_iter': n_iter,
        'n_filters': n_filters,
        'im_size': image_size,
        'dcomp': dcomp,
        'normalize_image': normalize_image,
        'refine_smaps': refine_smaps,
    }

    model = NCPDNet(**run_params)
    return evaluate_nc(
        model,
        multicoil=multicoil,
        dcomp=dcomp,
        three_d=three_d,
        **eval_kwargs,
    )

def evaluate_dcomp(multicoil=False, three_d=False, **eval_kwargs):
    if three_d:
        image_size = VOLUME_SIZE
    else:
        image_size = IM_SIZE
    model = NCDcompReconstructor(multicoil=multicoil, im_size=image_size)
    return evaluate_nc(
        model,
        multicoil=multicoil,
        dcomp=True,
        three_d=three_d,
        **eval_kwargs,
    )

def evaluate_unet(
        multicoil=False,
        n_layers=4,
        dcomp=False,
        base_n_filters=16,
        non_linearity='relu',
        **eval_kwargs
    ):
    run_params = {
        'non_linearity': non_linearity,
        'n_layers': n_layers,
        'layers_n_channels': [base_n_filters * 2**i for i in range(n_layers)],
        'layers_n_non_lins': 2,
        'res': True,
        'im_size': IM_SIZE,
        'dcomp': dcomp,
        'dealiasing_nc_fastmri': True,
        'multicoil': multicoil,
    }

    model = UnetComplex(**run_params)
    return evaluate_nc(
        model,
        multicoil=multicoil,
        dcomp=dcomp,
        **eval_kwargs,
    )

def evaluate_vnet(
        n_layers=4,
        dcomp=False,
        base_n_filters=16,
        non_linearity='relu',
        **eval_kwargs
    ):
    run_params = {
        'non_linearity': non_linearity,
        'layers_n_channels': [base_n_filters * 2**i for i in range(n_layers)],
        'layers_n_non_lins': 2,
        'res': True,
        'im_size': VOLUME_SIZE,
        'dcomp': dcomp,
        'dealiasing_nc': True,
    }

    model = VnetComplex(**run_params)
    eval_kwargs.update(three_d=True)
    return evaluate_nc(
        model,
        dcomp=dcomp,
        **eval_kwargs,
    )


def evaluate_nc_multinet(
        run_id=None,
        af=4,
        refine_smaps=False,
        multicoil=False,
        model='pdnet',
        acq_type='radial',
        n_epochs=200,
        n_samples=50,
        three_d=False,
        dcomp=False,
        n_filters=None,
        n_iter=10,
        normalize_image=False,
    ):
    if model == 'pdnet':
        evaluate_function = evaluate_ncpdnet
        if n_filters is None:
            n_filters = 32
        add_kwargs = {
            'refine_smaps': refine_smaps,
            'n_filters': n_filters,
            'n_iter': n_iter,
            'normalize_image': normalize_image,
        }
    elif model == 'unet':
        if n_filters is None:
            base_n_filters = 16
        else:
            base_n_filters = n_filters
        if three_d:
            evaluate_function = evaluate_vnet
        else:
            evaluate_function = evaluate_unet
        add_kwargs = {'base_n_filters': base_n_filters}
    if multicoil:
        add_kwargs.update(dcomp=True)
    else:
        add_kwargs.update(dcomp=dcomp)
    metric_names, metrics = evaluate_function(
        af=af,
        run_id=run_id,
        multicoil=multicoil,
        acq_type=acq_type,
        n_epochs=n_epochs,
        n_samples=n_samples,
        three_d=three_d,
        **add_kwargs,
    )
    return metric_names, metrics

@click.command()
@click.option(
    'af',
    '-a',
    type=int,
    default=4,
    help='The acceleration factor.'
)
@click.option(
    'run_id',
    '-r',
    type=str,
    default=None,
    help='The run id of the trained model.'
)
@click.option(
    'refine_smaps',
    '-rfs',
    is_flag=True,
    help='Whether you want to use an smaps refiner.'
)
@click.option(
    'multicoil',
    '-mc',
    is_flag=True,
    help='Whether you want to use multicoil data.'
)
@click.option(
    'model',
    '-m',
    type=str,
    default='pdnet',
    help='The NC model to use.'
)
@click.option(
    'acq_type',
    '-t',
    type=str,
    default='radial',
    help='The trajectory to use.'
)
@click.option(
    'n_epochs',
    '-e',
    type=int,
    default=200,
    help='The number of epochs during which the model was trained.'
)
@click.option(
    'n_samples',
    '-n',
    type=int,
    default=None,
    help='The number of samples to use for evaluation.'
)
@click.option(
    'three_d',
    '-3d',
    is_flag=True,
    help='Whether you want to use 3d data.'
)
@click.option(
    'dcomp',
    '-dc',
    is_flag=True,
    help='Whether you want to use density compensation.'
)
def evaluate_nc_click(
        af,
        run_id,
        refine_smaps,
        multicoil,
        model,
        acq_type,
        n_epochs,
        n_samples,
        three_d,
        dcomp,
    ):
    evaluate_nc_multinet(
        af=af,
        run_id=run_id,
        refine_smaps=refine_smaps,
        multicoil=multicoil,
        model=model,
        acq_type=acq_type,
        n_epochs=n_epochs,
        n_samples=n_samples,
        three_d=three_d,
        dcomp=dcomp,
    )


if __name__ == '__main__':
    evaluate_nc_click()
