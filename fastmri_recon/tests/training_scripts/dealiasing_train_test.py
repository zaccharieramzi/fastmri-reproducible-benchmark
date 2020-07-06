from fastmri_recon import config
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs


def test_train_dealiaser(create_full_fastmri_test_tmp_dataset):
    config.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    config.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    config.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    from fastmri_recon.training_scripts.dealias_train import train_dealiaser
    model_specs = get_model_specs(force_res=True, dealiasing=True)
    model_specs = [ms for ms in model_specs if ms[1] == 'small']
    for model_name, model_size, model_fun, kwargs, _, n_scales, _ in model_specs:
        train_dealiaser(
            model_fun=model_fun,
            model_kwargs=kwargs,
            n_scales=n_scales,
            run_id=f'{model_name}_{model_size}',
            n_epochs=1,
            n_steps_per_epoch=2,
        )
