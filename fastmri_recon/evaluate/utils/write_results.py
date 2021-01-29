import os.path as op
from pathlib import Path

import h5py

from fastmri_recon.config import FASTMRI_DATA_DIR


def _filename_submission(filename, use_v2=False):
    relative_filename = filename.split('/')[-1]
    name = relative_filename.split('.')[0]
    if 'v2' in name or not use_v2:
        name = relative_filename
    else:
        name += '_v2.h5'
    return name

def write_result(exp_id, result, filename, coiltype='multicoil', scale_factor=1e6, brain=False, challenge=False):
    """Write a reconstruction result to the correct file.

    Args:
        exp_id (str): the name of the experiments generating the reconstruction,
            typically the model name.
        result (ndarray): the numpy array containing the reconstruction.
        filename (str): the test or challenge filename from which the
            reconstruction was generated.
        coiltype (str): either `'singlecoil'`` or `'multicoil'``, indicating the type
            of acquisition. Defaults to `'multicoil'``.
        scale_factor (float): the scale factor used for the reconstruction.
            The result will be divided by this scale factor before being saved.
            Default to 1e6.
        brain (bool): whether the reconstruction is from the brain dataset.
            Defaults to False.
        challenge (bool): whether the reconstruction is from the challenge
            dataset. For now only applies for brain data. Defaults to False.
    """
    if brain:
        if challenge:
            res_main_dir = f'{FASTMRI_DATA_DIR}brain_{coiltype}_challenge/'
        else:
            res_main_dir = f'{FASTMRI_DATA_DIR}brain_{coiltype}_test/'
    else:
        res_main_dir = f'{FASTMRI_DATA_DIR}{coiltype}_test/'
    res_path = f'{res_main_dir}{exp_id}/'
    Path(res_path).mkdir(parents=True, exist_ok=True)
    res_formatted = result[..., 0] / scale_factor
    res_filename = _filename_submission(filename, use_v2=not brain or coiltype != 'multicoil')
    with h5py.File(op.join(res_path, res_filename), 'w') as f:
        f.create_dataset('reconstruction', data=res_formatted)
