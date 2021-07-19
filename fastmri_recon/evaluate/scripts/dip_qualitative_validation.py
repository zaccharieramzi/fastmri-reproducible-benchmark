from pathlib import Path
import time

import tensorflow as tf

from fastmri_recon.config import FASTMRI_DATA_DIR, OASIS_DATA_DIR, CHECKPOINTS_DIR
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.data.datasets.oasis_tf_records import train_nc_kspace_dataset_from_tfrecords as three_d_dataset
from fastmri_recon.data.datasets.multicoil.non_cartesian_tf_records import train_nc_kspace_dataset_from_tfrecords as multicoil_dataset
from fastmri_recon.evaluate.utils.save_figure import save_figure
from fastmri_recon.evaluate.reconstruction.dip_reconstruction import reconstruct_dip


IM_SIZE = (640, 400)
VOLUME_SIZE = (176, 256, 256)


def dip_qualitative_validation(
        model_kwargs,
        n_iter=1000,
        af=4,
        multicoil=False,
        three_d=False,
        acq_type='spiral',
        contrast=None,
        slice_index=15,
        brain=False,
        timing=False,
        zoom=None,
        draw_zoom=None,
    ):
    name = 'dip'
    if multicoil:
        name += '_mc'
        if brain:
            name += '_brain'
            val_path = f'{FASTMRI_DATA_DIR}brain_multicoil_val/'
            n_coils = None
        else:
            val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
            n_coils = 15
    elif three_d:
        name += '_3d'
        val_path = f'{OASIS_DATA_DIR}/val/'
        n_coils = 1
    else:
        val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'
        n_coils = 1
    model_kwargs.update(n_coils=n_coils)
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
        scale_factor=scale_factor,
        **add_kwargs
    )
    model_inputs, model_outputs = next(iter(val_set))
    if timing:
        start = time.time()
    model_path = f'dip_model_weights_{acq_type}'
    if contrast is not None:
        model_path += f'{contrast}'
    if multicoil:
        model_path += '_mc'
    if af is not None:
        model_path += f'_af{af}'
    if brain:
        n_coils = model_inputs[0].shape[1]
        output_shape = model_inputs[3][0]
        model_kwargs.update(n_coils=n_coils)
        model_path += f'_brain_{n_coils}'
    else:
        output_shape = (320, 320)
    model_path += '.h5'
    save_path = str(Path(CHECKPOINTS_DIR) / model_path)
    x = model_inputs[0:2]
    im_recos = reconstruct_dip(
        x[1],
        x[0],
        model_checkpoint=save_path,
        save_model=False,
        save_path=None,
        multicoil=multicoil,
        n_iter=n_iter,
        output_shape=output_shape,
        **model_kwargs,
    )
    if timing:
        end = time.time()
        duration = end - start
        print(f'Time for {name}, {acq_type}: {duration}')
    img_batch = model_outputs
    im_recos /= scale_factor
    img_batch /= scale_factor
    if not timing:
        save_figure(
            im_recos.numpy(),
            img_batch,
            name,
            slice_index=slice_index,
            af=af,
            three_d=three_d,
            acq_type=acq_type,
            zoom=zoom,
            draw_zoom=draw_zoom,
        )
    if timing:
        return duration
