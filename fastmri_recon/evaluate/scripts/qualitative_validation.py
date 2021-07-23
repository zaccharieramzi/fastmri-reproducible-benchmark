import time
from pathlib import Path

import tensorflow as tf
from tqdm.notebook import tqdm

from fastmri_recon.config import FASTMRI_DATA_DIR, CHECKPOINTS_DIR, OASIS_DATA_DIR
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.data.datasets.oasis_tf_records import train_nc_kspace_dataset_from_tfrecords as three_d_dataset
from fastmri_recon.data.datasets.multicoil.non_cartesian_tf_records import train_nc_kspace_dataset_from_tfrecords as multicoil_dataset
from fastmri_recon.evaluate.reconstruction.non_cartesian_dcomp_reconstruction import NCDcompReconstructor
from fastmri_recon.evaluate.utils.save_figure import save_figure
from fastmri_recon.models.subclassed_models.unet import UnetComplex
from fastmri_recon.models.subclassed_models.vnet import VnetComplex
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
from fastmri_recon.models.subclassed_models.pdnet import PDNet


tf.config.run_functions_eagerly(True)

IM_SIZE = (640, 400)
VOLUME_SIZE = (176, 256, 256)

def _extract_slice_of_batch(inputs, slice_index=15):
    if isinstance(inputs, (list, tuple)):
        return [_extract_slice_of_batch(i, slice_index) for i in inputs]
    else:
        return inputs[slice_index:slice_index+1]

def ncnet_qualitative_validation(
        model,
        name,
        run_id=None,
        n_epochs=100,
        af=4,
        multicoil=False,
        three_d=False,
        acq_type='spiral',
        gridding=False,
        contrast=None,
        dcomp=True,
        slice_index=15,
        brain=False,
        timing=False,
        zoom=None,
        draw_zoom=None,
    ):
    if multicoil:
        name += '_mc'
        if brain:
            name += '_brain'
            val_path = f'{FASTMRI_DATA_DIR}brain_multicoil_val/'
        else:
            val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
        if run_id is not None and acq_type not in run_id:
            name += '_rev'
    elif three_d:
        name += '_3d'
        val_path = f'{OASIS_DATA_DIR}/val/'
    else:
        val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'
    if multicoil:
        dataset = multicoil_dataset
        image_size = IM_SIZE
    elif three_d:
        dataset = three_d_dataset
        image_size = VOLUME_SIZE
    else:
        dataset = singlecoil_dataset
        image_size = IM_SIZE
    add_kwargs = {}
    if not multicoil and not three_d:
        add_kwargs.update(gridding=gridding)
    if multicoil:
        add_kwargs.update(brain=brain)
    if not three_d:
        add_kwargs.update(**{
            'contrast': contrast,
            'rand': False,
            'inner_slices': None,
        })
    scale_factor = 1e6 if not three_d else 1e-2
    val_set = dataset(
        val_path,
        image_size,
        af=af,
        acq_type=acq_type,
        compute_dcomp=dcomp,
        scale_factor=scale_factor,
        **add_kwargs
    )
    model_inputs, model_outputs = next(iter(val_set))
    model_inputs_dummy = _extract_slice_of_batch(
        model_inputs,
        0 if three_d else slice_index,
    )
    if run_id is not None:
        model.predict(model_inputs_dummy)
        chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5'
        model.load_weights(chkpt_path)
    if timing:
        if run_id is None:
            # to warm-up the graph
            model.predict(model_inputs_dummy)
        start = time.time()
    im_recos = model.predict(model_inputs, batch_size=8)
    if timing:
        end = time.time()
        duration = end - start
        print(f'Time for {name}, {acq_type}: {duration}')
    img_batch = model_outputs
    im_recos /= scale_factor
    img_batch /= scale_factor
    if not timing:
        save_figure(
            im_recos,
            img_batch,
            name,
            slice_index=slice_index,
            af=af,
            three_d=three_d,
            acq_type=acq_type,
            zoom=zoom,
            draw_zoom=draw_zoom,
            brain=brain,
        )
    if timing:
        return duration

def ncpdnet_qualitative_validation(
        multicoil=False,
        three_d=False,
        dcomp=False,
        normalize_image=False,
        n_iter=10,
        n_filters=32,
        n_primal=5,
        non_linearity='relu',
        refine_smaps=True,
        brain=False,
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
        'output_shape_spec': brain,
        'fastmri': not three_d,
    }

    model = NCPDNet(**run_params)
    name = 'pdnet'
    if dcomp:
        name += '-dcomp'
    if normalize_image:
        name += '-norm'
    return ncnet_qualitative_validation(
        model,
        name,
        multicoil=multicoil,
        dcomp=dcomp,
        three_d=three_d,
        brain=brain,
        **eval_kwargs,
    )

def gridded_pdnet_qualitative_validation(
        n_iter=10,
        n_filters=32,
        n_primal=5,
        non_linearity='relu',
        **eval_kwargs
    ):
    run_params = {
        'n_primal': n_primal,
        'activation': non_linearity,
        'n_iter': n_iter,
        'n_filters': n_filters,
        'primal_only': True,
    }
    model = PDNet(**run_params)
    eval_kwargs.update(dict(
        multicoil=False,
        dcomp=False,
        three_d=False,
        gridding=True,
    ))
    name = 'pdnet-gridded'
    return ncnet_qualitative_validation(
        model,
        name,
        **eval_kwargs,
    )

def dcomp_qualitative_validation(multicoil=False, three_d=False, brain=False, **eval_kwargs):
    if three_d:
        image_size = VOLUME_SIZE
    else:
        image_size = IM_SIZE
    model = NCDcompReconstructor(
        multicoil=multicoil,
        im_size=image_size,
        fastmri_format=not three_d,
        brain=brain,
    )
    name = 'adj-dcomp'
    eval_kwargs.update(dcomp=True)
    return ncnet_qualitative_validation(
        model,
        name,
        multicoil=multicoil,
        three_d=three_d,
        brain=brain,
        **eval_kwargs,
    )

def unet_qualitative_validation(
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
    name = 'unet'
    return ncnet_qualitative_validation(
        model,
        name,
        multicoil=multicoil,
        dcomp=dcomp,
        **eval_kwargs,
    )

def vnet_qualitative_validation(
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
    name = 'vnet'
    return ncnet_qualitative_validation(
        model,
        name,
        dcomp=dcomp,
        **eval_kwargs,
    )


def nc_multinet_qualitative_validation(
        run_id=None,
        af=4,
        refine_smaps=False,
        multicoil=False,
        model='pdnet',
        acq_type='radial',
        n_epochs=200,
        three_d=False,
        dcomp=False,
        n_filters=None,
        n_iter=10,
        normalize_image=False,
        slice_index=15,
        contrast=None,
        brain=False,
        timing=False,
        n_primal=5,
        zoom=None,
        draw_zoom=None,
    ):
    if model == 'pdnet':
        evaluate_function = ncpdnet_qualitative_validation
        if n_filters is None:
            n_filters = 32
        add_kwargs = {
            'refine_smaps': refine_smaps,
            'n_filters': n_filters,
            'n_iter': n_iter,
            'normalize_image': normalize_image,
            'n_primal': n_primal,
        }
    elif model == 'pdnet-gridded':
        evaluate_function = gridded_pdnet_qualitative_validation
        if n_filters is None:
            n_filters = 32
        add_kwargs = {
            'n_filters': n_filters,
            'n_iter': n_iter,
        }
    elif model == 'unet':
        if n_filters is None:
            base_n_filters = 16
        else:
            base_n_filters = n_filters
        if three_d:
            evaluate_function = vnet_qualitative_validation
        else:
            evaluate_function = unet_qualitative_validation
        add_kwargs = {'base_n_filters': base_n_filters}
    elif model == 'adj-dcomp':
        evaluate_function = dcomp_qualitative_validation
        add_kwargs = {}
    if multicoil:
        add_kwargs.update(dcomp=True)
    else:
        add_kwargs.update(dcomp=dcomp)
    return evaluate_function(
        af=af,
        run_id=run_id,
        multicoil=multicoil,
        acq_type=acq_type,
        n_epochs=n_epochs,
        three_d=three_d,
        slice_index=slice_index,
        contrast=contrast,
        brain=brain,
        timing=timing,
        zoom=zoom,
        draw_zoom=draw_zoom,
        **add_kwargs,
    )
