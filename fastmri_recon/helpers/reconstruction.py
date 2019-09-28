import numpy as np

from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold, LinearCompositionProx

from ..data.data_utils import from_test_file_to_mask_and_kspace
from .fourier import FFT2
from .utils import crop_center


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



def reco_wav(kspace, gradient_op, mu=1*1e-8, max_iter=10, nb_scales=4, wavelet_name='db4'):
    # for now this is only working with my fork of pysap-fastMRI
    # I will get it changed soon so that we don't need to ask for a specific
    # pysap-mri install
    from ..wavelets import WaveletDecimated
    from mri.numerics.reconstruct import sparse_rec_fista

    linear_op = WaveletDecimated(
        nb_scale=nb_scales,
        wavelet_name=wavelet_name,
        padding='periodization',
    )

    prox_op = LinearCompositionProx(
        linear_op=linear_op,
        prox_op=SparseThreshold(Identity(), None, thresh_type="soft"),
    )
    gradient_op.obs_data = kspace
    cost_op = None
    x_final, _, _, _ = sparse_rec_fista(
        gradient_op=gradient_op,
        linear_op=Identity(),
        prox_op=prox_op,
        cost_op=cost_op,
        xi_restart=0.96,
        s_greedy=1.1,
        mu=mu,
        restart_strategy='greedy',
        pov='analysis',
        max_nb_of_iter=max_iter,
        metrics=None,
        metric_call_period=1,
        verbose=0,
        progress=False,
    )
    x_final = np.abs(x_final)
    x_final = crop_center(x_final, 320)
    return x_final


def reco_iterative_from_test_file(filename, rec_type='wav', **kwargs):
    mask, kspaces = from_test_file_to_mask_and_kspace(filename)
    # mask handling
    fake_kspace = np.zeros_like(kspaces[0])
    fourier_mask = np.repeat(mask.astype(np.float)[None, :], fake_kspace.shape[0], axis=0)
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

def reco_and_gt_zfilled_from_val_file(kspace_and_mask_batch, img_batch, crop=True):
    kspaces, _ = kspace_and_mask_batch
    kspaces = np.squeeze(kspaces)
    im_recos = zero_filled_recon(kspaces, crop=crop)
    images = np.squeeze(img_batch)
    return im_recos, images


def reco_unet_from_test_file(zero_img_batch, means, stddevs, model):
    im_recos = model.predict_on_batch(zero_img_batch)
    im_recos = np.squeeze(im_recos)
    im_recos *= np.array(stddevs)[:, None, None]
    im_recos += np.array(means)[:, None, None]
    return im_recos

def reco_and_gt_unet_from_val_file(zero_img_batch, img_batch, means, stddevs, model):
    im_recos = reco_unet_from_test_file(zero_img_batch, means, stddevs, model)
    img_batch = np.squeeze(img_batch)
    img_batch *= np.array(stddevs)[:, None, None]
    img_batch += np.array(means)[:, None, None]
    return im_recos, img_batch

def reco_and_gt_unet_from_val_file_no_norm(zero_img_batch, img_batch, model):
    im_recos = model.predict_on_batch(zero_img_batch)
    im_recos = np.squeeze(im_recos)
    return im_recos, np.squeeze(img_batch)


def reco_net_from_test_file(kspace_and_mask_batch, model):
    im_recos = model.predict_on_batch(kspace_and_mask_batch)
    im_recos = np.squeeze(im_recos)
    return im_recos

def reco_and_gt_net_from_val_file(kspace_and_mask_batch, img_batch, model):
    im_recos = reco_net_from_test_file(kspace_and_mask_batch, model)
    img_batch = np.squeeze(img_batch)
    return im_recos, img_batch
