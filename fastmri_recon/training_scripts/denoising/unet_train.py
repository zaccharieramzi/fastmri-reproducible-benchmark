import time

from .generic_train import train_denoiser
from fastmri_recon.models.functional_models.unet import unet


def train_unet(name='big', train_kwargs=None, **model_kwargs):
    run_id = f'unet_denoising_{name}_{int(time.time())}'
    model = unet(
        input_size=(None, None, 1),
        n_output_channels=1,
        compile=False,
        **model_kwargs,
    )
    if train_kwargs is None:
        train_kwargs = {}
    return train_denoiser(model, run_id, **train_kwargs)
