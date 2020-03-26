import numpy as np

from ...data.utils.fourier import FFT2
from ...data.utils.crop import crop_center


def reco_z_filled(kspace, fourier_op):
    x_final = fourier_op.adj_op(kspace)
    x_final = np.abs(x_final)
    x_final = crop_center(x_final, 320)
    return x_final

def zero_filled_cropped_recon(kspace):
    """Perform a fastMRI zero-filled reconstruction on the kspace.

    This function performs an inverse fourier transform on the zero filled
    kspace, then takes the modulus of the result and crops it to fastMRI
    proportions.

    Parameters:
    kspace (ndarray): the zero-filled kspace

    Returns:
    ndarray: the image obtained by zero-filled reconstruction in fastMRI format
    """
    fourier_op = FFT2(np.ones_like(kspace))
    x_final = reco_z_filled(kspace, fourier_op)
    return x_final


def zero_filled_recon(kspaces, crop=False):
    """Perform a zero-filled reconstruction on a volume"""
    fourier_op = FFT2(np.ones_like(kspaces[0]))
    x_final = np.empty_like(kspaces)
    for i, kspace in enumerate(kspaces):
        x_final[i] = fourier_op.adj_op(kspace)
    x_final = np.abs(x_final)
    if crop:
        x_final_cropped = np.empty((len(kspaces), 320, 320))
        for i, x in enumerate(x_final):
            x_final_cropped[i] = crop_center(x, 320)
        x_final = x_final_cropped
    return x_final

def reco_and_gt_zfilled_from_val_file(kspace_and_mask_batch, img_batch, crop=True):
    kspaces, _ = kspace_and_mask_batch
    kspaces = np.squeeze(kspaces)
    im_recos = zero_filled_recon(kspaces, crop=crop)
    images = np.squeeze(img_batch)
    return im_recos, images
