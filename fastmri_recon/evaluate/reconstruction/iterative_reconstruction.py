import numpy as np

from .wavelet_reconstruction import reco_wav
from .zero_filled_reconstruction import reco_z_filled
from ...data.utils.fourier import FFT2 
from ...data.utils.h5 import from_test_file_to_mask_and_kspace


def reco_iterative_from_test_file(filename, rec_type='wav', **kwargs):
    mask, kspaces = from_test_file_to_mask_and_kspace(filename)
    # mask handling
    fake_kspace = np.zeros_like(kspaces[0])
    fourier_mask = np.repeat(mask.astype(np.float32)[None, :], fake_kspace.shape[0], axis=0)
    # op creation
    fourier_op_masked = FFT2(mask=fourier_mask)
    if rec_type == 'wav':
        from mri.numerics.gradient import GradAnalysis2
        gradient_op = GradAnalysis2(
            data=fake_kspace,
            fourier_op=fourier_op_masked,
        )
        im_recos = np.array([reco_wav(kspace * fourier_mask, gradient_op, **kwargs) for kspace in kspaces])
    elif rec_type == 'z_filled':
        im_recos = np.array([reco_z_filled(kspace * fourier_mask, fourier_op_masked) for kspace in kspaces])
    else:
        raise ValueError('{} not recognized as reconstruction type'.format(rec_type))
    return im_recos
