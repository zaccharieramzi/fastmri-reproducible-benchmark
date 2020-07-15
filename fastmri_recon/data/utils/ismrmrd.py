from pathlib import Path

import ismrmrd
from ismrmrd import Dataset, Acquisition, EncodingCounters
import numpy as np

from .h5 import _from_file_to_stuff
from .masking.gen_mask import gen_mask_equidistant


def kspace_to_ismrmrd(kspace, header, mask, file_index, out_dir='./'):
    header = ismrmrd.xsd.CreateFromDocument(header)
    header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum = kspace.shape[-1]
    header.encoding[0].encodingLimits.kspace_encoding_step_1.center = kspace.shape[-1] // 2
    header = header.toxml()
    n_slices = kspace.shape[0]
    for i_slice in range(n_slices):
        kspace_slice = kspace[i_slice]
        path = Path(out_dir) / f'{file_index}_slice_{i_slice}.h5'
        ds = ismrmrd.Dataset(path)
        ds.write_xml_header(header)
        for i_line, m in enumerate(np.squeeze(mask)):
            if m:
                acq = Acquisition.from_array(
                    kspace_slice[:, :, i_line],
                    idx=EncodingCounters(kspace_encode_step_1=i_line),
                    center_sample=320,
                )
                ds.append_acquisition(acq)

def from_fastmri_to_ismrmrd(filename, out_dir='./', accel_factor=4, split='val'):
    kspace, header = _from_file_to_stuff(filename, vals=['kspace', 'ismrmrd_header'])
    file_index = Path(filename).stem
    if split == 'test':
        raise NotImplementedError('test ismrmrd generation not implemented.')
    else:
        mask = gen_mask_equidistant(kspace, accel_factor=accel_factor)
    kspace_to_ismrmrd(kspace, header, mask, file_index, out_dir=out_dir)
