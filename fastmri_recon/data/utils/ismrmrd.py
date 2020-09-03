from pathlib import Path

import ismrmrd
from ismrmrd import Dataset, Acquisition, EncodingCounters
import numpy as np

from .h5 import _from_file_to_stuff
from .masking.gen_mask import gen_mask_equidistant


def get_flag_for_position(i_line, mask, accel_factor=4):
    num_cols = len(mask)
    center_fraction = (32 // accel_factor) / 100
    num_low_freqs = int(round(num_cols * center_fraction))
    acs_lim = (num_cols - num_low_freqs + 1) // 2
    if i_line < acs_lim or i_line >= acs_lim + num_low_freqs:
        return 0
    mask_offset = np.where(mask)[0][0] % accel_factor
    if i_line % accel_factor == mask_offset:
        return 21
    else:
        return 20

def kspace_to_ismrmrd(kspace, header, mask, file_index, out_dir='./', accel_factor=4, scale_factor=1e6):
    header = ismrmrd.xsd.CreateFromDocument(header)
    # TODO: this is only for 3T, to adapt to make sure we use the corect one
    header.experimentalConditions.H1resonanceFrequency_Hz = 128000000
    header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum = kspace.shape[-1]
    header.encoding[0].encodingLimits.kspace_encoding_step_1.center = kspace.shape[-1] // 2
    header.encoding[0].encodingLimits.kspace_encoding_step_2.maximum = 0
    header.encoding[0].encodingLimits.kspace_encoding_step_2.center = 0
    header.encoding[0].reconSpace = header.encoding[0].encodedSpace
    header.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1 = accel_factor
    header.encoding[0].parallelImaging.calibrationMode = 'embedded'
    header = header.toxml()
    n_slices = kspace.shape[0]
    for i_slice in range(n_slices):
        kspace_slice = kspace[i_slice] * scale_factor
        path = Path(out_dir) / f'{file_index}_slice_{i_slice:02d}.h5'
        ds = Dataset(path)
        ds.write_xml_header(header)
        for i_line, m in enumerate(np.squeeze(mask)):
            if m:
                flag = get_flag_for_position(i_line, np.squeeze(mask), accel_factor=accel_factor)
                acq = Acquisition.from_array(
                    kspace_slice[:, :, i_line],
                    idx=EncodingCounters(kspace_encode_step_1=i_line),
                    center_sample=kspace.shape[-2] // 2,
                )
                if flag:
                    acq.set_flag(flag)
                ds.append_acquisition(acq)

def from_fastmri_to_ismrmrd(filename, out_dir='./', accel_factor=4, split='val', scale_factor=1e6):
    kspace, header = _from_file_to_stuff(filename, vals=['kspace', 'ismrmrd_header'])
    file_index = Path(filename).stem
    if split == 'test':
        raise NotImplementedError('test ismrmrd generation not implemented.')
    else:
        mask = gen_mask_equidistant(kspace, accel_factor=accel_factor)
    kspace_to_ismrmrd(
        kspace,
        header,
        mask,
        file_index,
        out_dir=out_dir,
        accel_factor=accel_factor,
        scale_factor=scale_factor,
    )
