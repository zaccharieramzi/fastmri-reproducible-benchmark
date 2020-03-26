from fastmri_recon.models.functional_models.cascading import cascade_net


def test_init_cascading():
    run_params = {
        'n_cascade': 5,
        'n_convs': 5,
        'n_filters': 48,
        'noiseless': True,
    }
    model = cascade_net(lr=1e-3, **run_params)
