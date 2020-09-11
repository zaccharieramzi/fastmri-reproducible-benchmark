from fastmri_recon.evaluate.metrics.tf_metrics import *
from fastmri_recon.models.training.compile import compound_l1_mssim_loss


CUSTOM_TF_OBJECTS = {
    'keras_psnr': keras_psnr,
    'keras_ssim': keras_ssim,
    'compound_l1_mssim_loss': compound_l1_mssim_loss,
}
