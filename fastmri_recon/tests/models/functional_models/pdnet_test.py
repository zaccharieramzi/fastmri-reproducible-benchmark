from fastmri_recon.models.functional_models.pdnet import pdnet


def test_init_pdnet():
    run_params = {
        'n_primal': 5,
        'n_dual': 5,
        'n_iter': 10,
        'n_filters': 32,
    }
    model = pdnet(lr=1e-3, **run_params)
