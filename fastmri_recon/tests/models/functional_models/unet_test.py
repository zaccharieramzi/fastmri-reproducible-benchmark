from fastmri_recon.models.functional_models.unet import unet


def test_init_unet():
    run_params = {
        'n_layers': 4,
        'pool': 'max',
        "layers_n_channels": [16, 32, 64, 128],
        'layers_n_non_lins': 2,
    }
    model = unet(input_size=(320, 320, 1), lr=1e-3, **run_params)
