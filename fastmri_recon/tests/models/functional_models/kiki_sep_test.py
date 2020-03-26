from fastmri_recon.models.functional_models.kiki_sep import kiki_sep_net
from fastmri_recon.models.utils.data_consistency import MultiplyScalar
from fastmri_recon.models.utils.non_linearities import lrelu


def test_init_kiki_sep():
    run_params = {
        'n_convs': 25,
        'n_filters': 48,
        'noiseless': True,
        'lr': 1e-3,
        'activation': lrelu,
    }
    multiply_scalar = MultiplyScalar()
    model = kiki_sep_net(None, multiply_scalar, to_add='K', last=False, **run_params)
    model = kiki_sep_net(model, multiply_scalar, to_add='I', last=False, **run_params)
    model = kiki_sep_net(model, multiply_scalar, to_add='K', last=False, **run_params)
    model = kiki_sep_net(model, multiply_scalar, to_add='I', last=True, **run_params)
