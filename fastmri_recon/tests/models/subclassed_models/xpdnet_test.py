import pytest
import tensorflow as tf
from tensorflow.keras import mixed_precision

from fastmri_recon.models.subclassed_models.denoisers.mwcnn import MWCNN
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet


@pytest.mark.parametrize('primal_only, n_dual', [
    (False, 1),
    (True, 2),
])
@pytest.mark.parametrize('use_mixed_precision', [True, False])
def test_xpdnet(primal_only, n_dual, use_mixed_precision):
    if use_mixed_precision:
        policy_type = 'mixed_float16'
    else:
        policy_type = 'float32'
    policy = mixed_precision.Policy(policy_type)
    mixed_precision.set_global_policy(policy)
    n_primal = 2
    n_scales = 3
    submodel_kwargs = dict(
        n_scales=n_scales,
        kernel_size=3,
        bn=False,
        n_filters_per_scale=[4, 8, 8],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=2,
        first_conv_n_filters=4,
        res=False,
        n_outputs=2*n_primal,
    )
    model = XPDNet(
        model_fun=MWCNN,
        model_kwargs=submodel_kwargs,
        n_primal=n_primal,
        n_iter=2,
        multicoil=True,
        n_scales=n_scales,
        primal_only=primal_only,
        n_dual=n_dual,
    )
    model([
        tf.zeros([1, 5, 640, 320, 1], dtype=tf.complex64),  # kspace
        tf.zeros([1, 5, 640, 320], dtype=tf.complex64),  # mask
        tf.zeros([1, 5, 640, 320], dtype=tf.complex64),  # smaps
    ])
