from pathlib import Path

from fastmri_recon.config import FASTMRI_DATA_DIR
from fastmri_recon.data.utils.ismrmrd import from_fastmri_to_ismrmrd


if __name__ == '__main__':
    af = 4
    out = f'./val_knee_{af}/'
    fastmri_path = Path(FASTMRI_DATA_DIR) / 'multicoil_val'
    filenames = fastmri_path.glob('*.h5')
    for f in filenames:
        from_fastmri_to_ismrmrd(filename, out_dir='./', accel_factor=af)
