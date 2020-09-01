import numpy as np
from pygrappa import mdgrappa

from ...data.utils.crop import crop_center
from ...data.utils.fourier import ifft


def reco_grappa_kspace(kspace, af=4, **grappa_kwargs):
    n_slices, _, _, sy = kspace.shape[:]
    if af == 4:
        n_acs = int(sy * 8 / 100)
    else:
        n_acs = int(sy * 4 / 100)
    ctr, pd = sy // 2, n_acs // 2
    calib = kspace[..., ctr-pd:ctr+pd].copy() # call copy()!
    recon = np.empty_like(kspace)
    for i in range(n_slices):
        recon[i] = mdgrappa(
            kspace[i].astype(np.complex),
            calib[i].astype(np.complex),
            coil_axis=0,
            **grappa_kwargs,
        )
    return recon

def reco_grappa(kspace, af=4, **grappa_kwargs):
    recon = reco_grappa_kspace(kspace, af=af, **grappa_kwargs)
    recon = ifft(recon)
    x_final = np.linalg.norm(recon, axis=1)
    x_final = crop_center(x_final, 320)
    return x_final
