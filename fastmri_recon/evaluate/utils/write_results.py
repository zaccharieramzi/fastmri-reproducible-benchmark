import os.path as op
from pathlib import Path

import h5py

from fastmri_recon.config import FASTMRI_DATA_DIR


def _filename_submission(filename):
    relative_filename = filename.split('/')[-1]
    name = relative_filename.split('.')[0]
    if 'v2' in name:
        name = relative_filename
    else:
        name += '_v2.h5'
    return name

def write_result(exp_id, result, filename, coiltype='multicoil', scale_factor=1e6, brain=False):
    if brain:
        res_main_dir = f'{FASTMRI_DATA_DIR}brain_{coiltype}_test/
    else:
        res_main_dir = f'{FASTMRI_DATA_DIR}{coiltype}_test/
    res_path = f'{res_main_dir}{exp_id}/'
    Path(res_path).mkdir(parents=True, exist_ok=True)
    res_formatted = result[..., 0] / scale_factor
    res_filename = _filename_submission(filename)
    with h5py.File(op.join(res_path, res_filename), 'w') as f:
        f.create_dataset('reconstruction', data=res_formatted)
