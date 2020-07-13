from pathlib import Path

from ismrmrd import Dataset, Acquisition, EncodingCounters

from .h5 import _from_file_to_stuff


def kspace_to_ismrmrd(kspace, header, mask, file_index):
    n_slices = kspace.shape[0]
    for i_slice in range(n_slices):
        kspace_slice = kspace[i_slice]
        ds = ismrmrd.Dataset(f'{file_index}_slice_{i_slice}.h5')
        ds.write_xml_header(header)
        for i_line, m in enumerate(np.squeeze(mask)):
            if m:
                acq = Acquisition.from_array(
                    start_kspace[:, i_line, :],
                    idx=EncodingCounters(kspace_encode_step_1=i_line),
                )
                ds.append_acquisition(acq)

def from_fastmri_to_ismrmrd(filename):
    kspace, header = _from_file_to_stuff(filename, vals=['kspace', 'ismrmrd_header'])
    file_index = Path(filename).stem
    mask = gen_mask_equidistant(kspace)
    kspace_to_ismrmrd(kspace, header, mask, file_index)
