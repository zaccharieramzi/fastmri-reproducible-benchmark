from fastmri_recon.models.subclassed_models.denoisers.dncnn import DnCNN
from fastmri_recon.training_scripts import postprocess_train, xpdnet_train
from fastmri_recon.training_scripts.postprocess_train import train_vnet_postproc
from fastmri_recon.training_scripts.xpdnet_train import train_xpdnet


n_primal = 2

def test_train_vnet_postproc(create_full_fastmri_test_tmp_dataset):
    xpdnet_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    xpdnet_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    xpdnet_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    xpdnet_train.n_volumes_train = 2
    postprocess_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    postprocess_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    postprocess_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    postprocess_train.n_volumes_train = 2
    orig_model_fun = DnCNN
    orig_model_kwargs = dict(n_convs=2, n_filters=4, res=False, n_outputs=2*n_primal)
    n_scales = 0
    res = True
    original_run_id = train_xpdnet(
        model_fun=orig_model_fun,
        model_kwargs=orig_model_kwargs,
        n_scales=n_scales,
        n_primal=n_primal,
        res=res,
        multicoil=False,
        af=1,
        n_samples=2,
        n_epochs=1,
        n_iter=1,
    )
    train_vnet_postproc(
        orig_model_fun=orig_model_fun,
        orig_model_kwargs=orig_model_kwargs,
        original_run_id=original_run_id,
        n_scales=n_scales,
        n_primal=n_primal,
        res=res,
        multicoil=False,
        af=1,
        n_samples=2,
        n_epochs=1,
        n_epochs_original=1,
        n_iter=1,
    )
