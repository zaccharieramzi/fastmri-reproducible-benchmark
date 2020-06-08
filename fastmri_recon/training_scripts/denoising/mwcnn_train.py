import time

from .generic_train import train_denoiser
from fastmri_recon.models.subclassed_models.denoisers.mwcnn import MWCNN


def train_mwcnn(name='big', train_kwargs=None, **model_kwargs):
    run_id = f'mwcnn_denoising_{name}_{int(time.time())}'
    model = MWCNN(**model_kwargs)
    if train_kwargs is None:
        train_kwargs = {}
    return train_denoiser(model, run_id, **train_kwargs)
