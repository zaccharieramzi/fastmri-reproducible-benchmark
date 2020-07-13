from pathlib import Path

import click

from fastmri_recon.config import FASTMRI_DATA_DIR
from fastmri_recon.data.utils.ismrmrd import from_fastmri_to_ismrmrd


def generate_ismrmrd(af=4, split='val', organ='knee'):
    original_directory = f'multicoil_{split}'
    if organ == 'brain':
        original_directory = 'brain_' + original_directory
    out = f'./{split}_{organ}_{af}/'
    fastmri_path = Path(FASTMRI_DATA_DIR) / original_directory
    filenames = fastmri_path.glob('*.h5')
    for f in filenames:
        from_fastmri_to_ismrmrd(filename, out_dir=out, accel_factor=af, split=split)
