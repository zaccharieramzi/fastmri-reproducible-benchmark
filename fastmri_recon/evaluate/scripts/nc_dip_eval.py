import os
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.data.datasets.oasis_tf_records import train_nc_kspace_dataset_from_tfrecords as three_d_dataset
from fastmri_recon.data.datasets.multicoil.non_cartesian_tf_records import train_nc_kspace_dataset_from_tfrecords as multicoil_dataset
from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from fastmri_recon.evaluate.reconstruction.dip_reconstruction import reconstruct_dip


# this number means that 99.56% of all images will not be affected by
# cropping
IM_SIZE = (640, 400)
VOLUME_SIZE = (176, 256, 256)


def evaluate_dip_nc(
        model_kwargs,
        multicoil=False,
        three_d=False,
        n_iter=10_000,
        contrast=None,
        acq_type='radial',
        n_samples=None,
        brain=False,
        **acq_kwargs,
    ):
    if multicoil:
        if brain:
            val_path = f'{FASTMRI_DATA_DIR}brain_multicoil_val/'
            n_coils = None
        else:
            val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
            n_coils = 15
    elif three_d:
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
    if not three_d:
        add_kwargs = {
            'contrast': contrast,
            'rand': False,
            'inner_slices': None,
        }
        if multicoil:
            add_kwargs.update(brain=brain)
    else:
        add_kwargs = {}
    add_kwargs.update(**acq_kwargs)
    val_set = dataset(
        val_path,
        image_size,
        acq_type=acq_type,
        compute_dcomp=False,
        scale_factor=1e6 if not three_d else 1e-2,
        **add_kwargs
    )
    if n_samples is not None:
        val_set = val_set.take(n_samples)
    else:
        val_set = val_set.take(199)

    res_name = f'dip_eval_on_{acq_type}'
    if multicoil:
        res_name += '_mc'
        if brain:
            res_name += '_brain'
    if three_d:
        res_name += '_3d'
    if contrast is not None:
        res_name += f'_{contrast}'
    if acq_kwargs:
        af = acq_kwargs['af']
        if af != 4:
            res_name += f'_af{af}'
    else:
        af = None
    if three_d:
        m = Metrics({'PSNR': METRIC_FUNCS['PSNR']}, res_name)
    else:
        m = Metrics(METRIC_FUNCS, res_name)
    model_path = f'dip_model_weights_{acq_type}'
    if contrast is not None:
        model_path += f'{contrast}'
    if multicoil:
        model_path += '_mc'
    if af is not None:
        model_path += f'_af{af}'
    if brain:
        model_checkpoints = {}
        model_path += '_brain_{n_coils}.h5'
    else:
        model_checkpoint = None
        model_path += '.h5'
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=199 if n_samples is None else n_samples):
        if brain:
            output_shape = x[3][0]
            n_coils = x[0].shape[1]
            save_path = str(Path(CHECKPOINTS_DIR) / model_path.format(n_coils=n_coils))
            model_checkpoint = model_checkpoints.get(n_coils, None)
            model_kwargs.update(n_coils=n_coils)
        else:
            output_shape = (320, 320)
            save_path = str(Path(CHECKPOINTS_DIR) / model_path)
        x = x[0:2]
        y_pred = reconstruct_dip(
            x[1],
            x[0],
            model_checkpoint=model_checkpoint,
            save_model=model_checkpoint is None,
            save_path=save_path,
            multicoil=multicoil,
            n_iter=n_iter if model_checkpoint is None else n_iter//10,
            output_shape=output_shape,
            **model_kwargs,
        )
        if brain:
            model_checkpoints[n_coils] = save_path
        else:
            model_checkpoint = save_path
        m.push(y_true[..., 0], y_pred[..., 0].numpy())
        del x
        del y_true
        del y_pred
    print(METRIC_FUNCS.keys())
    print(list(m.means().values()))
    m.to_csv()
    return METRIC_FUNCS, list(m.means().values())
