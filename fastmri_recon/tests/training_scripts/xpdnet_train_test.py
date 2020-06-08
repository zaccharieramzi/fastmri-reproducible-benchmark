import pytest

from fastmri_recon.models.subclassed_models.denoisers.dncnn import DnCNN
from fastmri_recon.models.subclassed_models.denoisers.focnet import FocNet
from fastmri_recon.models.subclassed_models.denoisers.mwcnn import MWCNN
from fastmri_recon.models.functional_models.unet import unet
from fastmri_recon.training_scripts import xpdnet_train
from fastmri_recon.training_scripts.xpdnet_train import train_xpdnet


n_primal = 2

@pytest.mark.parametrize('model, n_scales, res', [
    # dncnn
    (DnCNN(n_convs=2, n_filters=4, res=False, n_outputs=2*n_primal), 0, True),
    # focnet
])
def test_train_xpdnet(create_full_fastmri_test_tmp_dataset, model, n_scales, res):
    xpdnet_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    xpdnet_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    xpdnet_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    xpdnet_train.n_volumes_train = 2
    train_xpdnet(
        model=model,
        n_scales=n_scales,
        n_primal=n_primal,
        res=res,
        af=1,
        n_samples=2,
        n_epochs=1,
        n_iter=1,
    )
