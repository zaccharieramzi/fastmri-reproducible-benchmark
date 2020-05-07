import os.path as op
from pathlib import Path

import h5py

from fastmri_recon.config import FASTMRI_DATA_DIR


def _filename_submission(filename):
    relative_filename = filename.split('/')[-1]
    name = relative_filename.split('.')[0]
    name += '_v2.h5'
    return name

def write_result(run_id, result, filename, coiltype='multicoil', scale_factor=1e6):
    res_path = f'{FASTMRI_DATA_DIR}{coiltype}_test/{run_id}/'
    Path(res_path).mkdir(parents=True, exist_ok=True)
    res_formatted = result[..., 0] / scale_factor
    res_filename = _filename_submission(filename)
    with h5py.File(op.join(res_path, res_filename), 'w') as f:
        f.create_dataset('reconstruction', data=res_formatted)
