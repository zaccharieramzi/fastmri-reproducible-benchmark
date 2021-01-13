import time

try:
    from tf_fastmri_data import config as config_data
except ModuleNotFoundError:
    tf_fastmri_data_not_avail = True
else:
    tf_fastmri_data_not_avail = False
import pytest

from fastmri_recon import config as config_output
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs


@pytest.mark.skipif(tf_fastmri_data_not_avail, reason='tf-fastmri-data not installed')
def test_train_denoiser(create_full_fastmri_test_tmp_dataset):
    config_data.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    config_data.PATHS_MAP[False][False]['train'] = 'singlecoil_train/singlecoil_train'
    config_output.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    config_output.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    from fastmri_recon.training_scripts.denoising.generic_train import train_denoiser
    model_specs = get_model_specs(force_res=True)
    model_specs = [ms for ms in model_specs if ms[1] == 'small']
    for model_name, model_size, model_fun, kwargs, n_inputs, _, _ in model_specs:
        train_denoiser(
            model=(model_fun, kwargs, n_inputs),
            run_id=f'{model_name}_{model_size}_{int(time.time())}',
            n_epochs=1,
            n_steps_per_epoch=2,
        )
