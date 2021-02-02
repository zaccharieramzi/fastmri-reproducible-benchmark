import pytest

from fastmri_recon.models.functional_models.unet import unet
from fastmri_recon.models.subclassed_models.denoisers.didn import DIDN
from fastmri_recon.models.subclassed_models.denoisers.dncnn import DnCNN
from fastmri_recon.models.subclassed_models.denoisers.focnet import FocNet
from fastmri_recon.models.subclassed_models.denoisers.mwcnn import MWCNN
from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.unet import UnetMultiDomain
from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.mwcnn import MWCNNMultiDomain
from fastmri_recon.training_scripts import xpdnet_train
from fastmri_recon.training_scripts.xpdnet_train import train_xpdnet


n_primal = 2

@pytest.mark.parametrize('model_fun, model_kwargs, n_scales, res', [
    # didn
    (DIDN, dict(
        n_scales=3,
        n_filters=4,
        n_dubs=2,
        n_convs_recon=2,
        res=False,
        n_outputs=2*n_primal,
    ), 4, True),
    # dncnn
    (DnCNN, dict(n_convs=2, n_filters=4, res=False, n_outputs=2*n_primal), 0, True),
    # focnet
    (FocNet, dict(n_filters=4, n_outputs=2*n_primal), 4, False),
    # mwcnn
    (MWCNN, dict(
        n_scales=3,
        kernel_size=3,
        bn=False,
        n_filters_per_scale=[4, 8, 8],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=2,
        first_conv_n_filters=4,
        res=False,
        n_outputs=2*n_primal,
    ), 4, True),
    # unet
    (unet, dict(
        n_layers=2,
        layers_n_channels=[16, 32,],
        res=False,
        layers_n_non_lins=1,
        n_output_channels=2*n_primal,
        input_size=(None, None, 2*(n_primal + 1)),
    ), 2, True),
])
def test_train_xpdnet(create_full_fastmri_test_tmp_dataset, model_fun, model_kwargs, n_scales, res):
    xpdnet_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    xpdnet_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    xpdnet_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    xpdnet_train.n_volumes_train = 2
    train_xpdnet(
        model_fun=model_fun,
        model_kwargs=model_kwargs,
        n_scales=n_scales,
        n_primal=n_primal,
        res=res,
        multicoil=False,
        af=1,
        n_samples=2,
        n_epochs=1,
        n_iter=1,
    )

@pytest.mark.parametrize('model_fun, model_kwargs, n_scales, res', [
    # mwcnn
    (MWCNNMultiDomain, dict(
        n_scales=3,
        kernel_size=3,
        n_filters_per_scale=[4, 8, 8],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=2,
        first_conv_n_filters=4,
        res=False,
        n_outputs=2*n_primal,
    ), 4, True),
    # unet
    (UnetMultiDomain, dict(
        layers_n_channels=[16, 32,],
        layers_n_non_lins=1,
        n_outputs=2*n_primal,
    ), 2, True),
])
def test_train_xpdnet_flmd(create_full_fastmri_test_tmp_dataset, model_fun, model_kwargs, n_scales, res):
    xpdnet_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    xpdnet_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    xpdnet_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    xpdnet_train.n_volumes_train = 2
    train_xpdnet(
        model_fun=model_fun,
        model_kwargs=model_kwargs,
        n_scales=n_scales,
        n_primal=n_primal,
        res=res,
        multicoil=False,
        af=1,
        n_samples=2,
        n_epochs=1,
        n_iter=1,
    )

@pytest.mark.parametrize('multiscale_kspace_learning', [True, False])
def test_train_xpdnet_dual(create_full_fastmri_test_tmp_dataset, multiscale_kspace_learning):
    xpdnet_train.FASTMRI_DATA_DIR = create_full_fastmri_test_tmp_dataset['fastmri_tmp_data_dir']
    xpdnet_train.LOGS_DIR = create_full_fastmri_test_tmp_dataset['logs_tmp_dir']
    xpdnet_train.CHECKPOINTS_DIR = create_full_fastmri_test_tmp_dataset['checkpoints_tmp_dir']
    xpdnet_train.n_volumes_train = 2
    train_xpdnet(
        model_fun=MWCNN,
        model_kwargs=dict(
            n_scales=3,
            kernel_size=3,
            bn=False,
            n_filters_per_scale=[4, 8, 8],
            n_convs_per_scale=[2, 2, 2],
            n_first_convs=2,
            first_conv_n_filters=4,
            res=False,
            n_outputs=2*n_primal,
        ),
        n_scales=4,
        n_primal=n_primal,
        res=True,
        multicoil=True,
        af=1,
        n_samples=2,
        n_epochs=1,
        n_iter=1,
        primal_only=False,
        n_dual=2,
        n_dual_filters=4,
        multiscale_kspace_learning=multiscale_kspace_learning,
    )
