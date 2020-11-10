import pytest
import tensorflow as tf
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf

from fastmri_recon.data.utils.non_cartesian import get_stacks_of_radial_trajectory
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet


def test_ncpdnet_init_and_call(ktraj):
    model = NCPDNet(n_iter=3, n_primal=3, n_filters=8)
    image_shape = (640, 372)
    nspokes = 15
    traj = ktraj(image_shape, nspokes)
    spokelength = image_shape[-1] * 2
    kspace_shape = spokelength * nspokes
    model([
        tf.zeros([1, 1, kspace_shape, 1], dtype=tf.complex64),
        traj,
        (tf.constant([image_shape[-1]]),),
    ])

def test_ncpdnet_init_and_call_multicoil(ktraj):
    model = NCPDNet(n_iter=3, n_primal=3, n_filters=8, multicoil=True)
    image_shape = (640, 372)
    nspokes = 15
    traj = ktraj(image_shape, nspokes)
    spokelength = image_shape[-1] * 2
    kspace_shape = spokelength * nspokes
    model([
        tf.zeros([1, 1, kspace_shape, 1], dtype=tf.complex64),
        traj,
        tf.zeros([1, 1, *image_shape], dtype=tf.complex64),
        (tf.constant([image_shape[-1]]),),
    ])

@pytest.mark.parametrize('dcomp, volume_shape', [
    (True, (176, 32, 32)),
    (True, (32, 32, 32)),
    (False, (176, 32, 32)),
])
def test_ncpdnet_init_and_call_3d(dcomp, volume_shape):
    model = NCPDNet(
        n_iter=1,
        n_primal=2,
        n_filters=2,
        multicoil=False,
        im_size=volume_shape,
        three_d=True,
        dcomp=dcomp,
        fastmri=False,
    )
    af = 16
    traj = get_stacks_of_radial_trajectory(volume_shape, af=af)
    spokelength = volume_shape[-2]
    nspokes = volume_shape[-1] // af
    nstacks = volume_shape[0]
    kspace_shape = nspokes*spokelength*nstacks
    extra_args = (tf.constant([volume_shape]),)
    if dcomp:
        nufft_ob = KbNufftModule(
            im_size=volume_shape,
            grid_size=None,
            norm='ortho',
        )
        interpob = nufft_ob._extract_nufft_interpob()
        nufftob_forw = kbnufft_forward(interpob)
        nufftob_back = kbnufft_adjoint(interpob)
        dcomp = calculate_radial_dcomp_tf(
            interpob,
            nufftob_forw,
            nufftob_back,
            traj[0],
            stacks=True,
        )
        dcomp = tf.ones([1, tf.shape(dcomp)[0]], dtype=dcomp.dtype) * dcomp[None, :]
        extra_args += (dcomp,)
    res = model([
        tf.zeros([1, 1, kspace_shape, 1], dtype=tf.complex64),
        traj,
        extra_args,
    ])
    assert res.shape[1:4] == volume_shape
