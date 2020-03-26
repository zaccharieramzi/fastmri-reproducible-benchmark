import numpy as np

from ...data.utils.crop import crop_center


def reco_wav(kspace, gradient_op, mu=1*1e-8, max_iter=10, nb_scales=4, wavelet_name='db4'):
    # for now this is only working with my fork of pysap-fastMRI
    # I will get it changed soon so that we don't need to ask for a specific
    # pysap-mri install
    from modopt.opt.linear import Identity
    from modopt.opt.proximity import SparseThreshold, LinearCompositionProx
    from mri.numerics.reconstruct import sparse_rec_fista

    from ...wavelets import WaveletDecimated

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
