import os

import click
import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as multicoil_dataset
from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from fastmri_recon.evaluate.reconstruction.non_cartesian_dcomp_reconstruction import NCDcompReconstructor
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
from fastmri_recon.models.subclassed_models.unet import UnetComplex


# this number means that 99.56% of all images will not be affected by
# cropping
IM_SIZE = (640, 400)

# TODO: replace these stupid functions with just extraction of the first slice...
def _extract_inputs_shape(inputs, no_batch=True):
    if isinstance(inputs, (list, tuple)):
        return [_extract_inputs_shape(i, no_batch=no_batch) for i in inputs]
    else:
        if no_batch:
            return [1] + inputs.shape[1:]
        else:
            return inputs.shape

def _extract_inputs_dtype(inputs):
    if isinstance(inputs, (list, tuple)):
        return [_extract_inputs_dtype(i) for i in inputs]
    else:
        return inputs.dtype

def _zeros_from_shape(shapes, dtypes):
    if isinstance(shapes, (list, tuple)):
        return [_zeros_from_shape(s, d) for s, d in zip(shapes, dtypes)]
    else:
        return tf.zeros(shapes, dtype=dtypes)

def evaluate_nc(
        model,
        multicoil=False,
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
    else:
        val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    if multicoil:
        dataset = multicoil_dataset
    else:
        dataset = singlecoil_dataset
    val_set = dataset(
        val_path,
        IM_SIZE,
        acq_type=acq_type,
        compute_dcomp=dcomp,
        contrast=contrast,
        inner_slices=None,
        rand=False,
        scale_factor=1e6,
        **acq_kwargs
    )
    if n_samples is not None:
        val_set = val_set.take(n_samples)
    else:
        val_set = val_set.take(199)

    example_input = next(iter(val_set))[0]
    inputs_shape = _extract_inputs_shape(example_input, no_batch=True)
    inputs_dtype = _extract_inputs_dtype(example_input)

    inputs = _zeros_from_shape(inputs_shape, inputs_dtype)
    # special case for the shape:
    inputs[-1][0] = tf.constant([[372]])
    model(inputs)
    if run_id is not None:
        model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    m = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=199 if n_samples is None else n_samples):
        y_pred = model.predict(x, batch_size=1)
        m.push(y_true[..., 0], y_pred[..., 0])
    return METRIC_FUNCS, list(m.means().values())

def evaluate_ncpdnet(
        multicoil=False,
        dcomp=False,
        normalize_image=False,
        n_iter=10,
        n_filters=32,
        n_primal=5,
        non_linearity='relu',
        refine_smaps=True,
        **eval_kwargs
    ):
    run_params = {
        'n_primal': n_primal,
        'multicoil': multicoil,
        'activation': non_linearity,
        'n_iter': n_iter,
        'n_filters': n_filters,
        'im_size': IM_SIZE,
        'dcomp': dcomp,
        'normalize_image': normalize_image,
        'refine_smaps': refine_smaps,
    }

    model = NCPDNet(**run_params)
    return evaluate_nc(
        model,
        multicoil=multicoil,
        dcomp=dcomp,
        **eval_kwargs,
    )

def evaluate_dcomp(multicoil=False, **eval_kwargs):
    model = NCDcompReconstructor(multicoil=multicoil, im_size=IM_SIZE)
    return evaluate_nc(
        model,
        multicoil=multicoil,
        dcomp=True,
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
    help='The number of epochs during which the model was trained.'
)
def evaluate_nc_click(
        af,
        run_id,
        refine_smaps,
        multicoil,
        model,
        acq_type,
        n_epochs,
    ):
    if model == 'pdnet':
        evaluate_function = evaluate_ncpdnet
        add_kwargs = {'refine_smaps': refine_smaps}
    elif model == 'unet':
        evaluate_function = evaluate_unet
        add_kwargs = {}
    if multicoil:
        add_kwargs.update(dcomp=True)
    evaluate_function(
        af=af,
        run_id=run_id,
        multicoil=multicoil,
        acq_type=acq_type,
        n_epochs=n_epochs,
        **add_kwargs,
    )


if __name__ == '__main__':
    evaluate_nc_click()
