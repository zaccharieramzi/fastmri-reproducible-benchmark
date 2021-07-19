import os
from pathlib import Path

import pandas as pd
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


def debug_dip_nc(
        model_kwargs,
        slice_index=15,
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
        else:
            val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
    elif three_d:
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

    if three_d:
        m = Metrics({'PSNR': METRIC_FUNCS['PSNR']})
    else:
        m = Metrics(METRIC_FUNCS)
    model_checkpoint = None
    model_path = f'dip_model_weights_{acq_type}'
    if contrast is not None:
        model_path += f'{contrast}'
    if multicoil:
        model_path += '_mc'
    if acq_kwargs:
        af = acq_kwargs['af']
        model_path += f'_af{af}'        
    model_path += '.h5'
    save_path = str(Path(CHECKPOINTS_DIR) / model_path)
    x, y_true = next(val_set.as_numpy_iterator())
    x = x[0:2]
    y_pred, _history = reconstruct_dip(
        x[1][slice_index:slice_index+1],
        x[0][slice_index:slice_index+1],
        model_checkpoint=None,
        save_model=True,
        save_path=save_path,
        multicoil=multicoil,
        n_iter=n_iter,
        debug=True,
        **model_kwargs,
    )
    print(save_path)
    pd.DataFrame(_history.history).to_csv("history.csv")
    return save_path
