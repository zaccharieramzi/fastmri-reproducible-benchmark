from functools import partial

from fastmri_recon.evaluate.metrics.tf_metrics import *
from fastmri_recon.models.training.compile import compound_l1_mssim_loss
from fastmri_recon.models.subclassed_models.vnet import Conv

mssim = partial(compound_l1_mssim_loss, alpha=0.9999)
mssim.__name__ = "mssim"
CUSTOM_TF_OBJECTS = {
    'keras_psnr': keras_psnr,
    'keras_ssim': keras_ssim,
    'compound_l1_mssim_loss': compound_l1_mssim_loss,
    'mssim': mssim,
    'Conv': Conv,
}
