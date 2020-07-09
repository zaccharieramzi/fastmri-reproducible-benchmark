import tensorflow as tf
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf

from ..utils.non_cartesian import get_radial_trajectory, get_debugging_cartesian_trajectory, get_spiral_trajectory
from fastmri_recon.models.utils.fourier import tf_unmasked_adj_op, nufft


def non_cartesian_from_volume_to_nc_kspace_and_traj(nfft_ob, volume_size, acq_type='radial', scale_factor=1, compute_dcomp=False, **acq_kwargs):
    def from_volume_to_nc_kspace_and_traj(volume):
        if acq_type == 'radial':
            traj = get_radial_trajectory(volume_size, **acq_kwargs)
        elif acq_type == 'cartesian':
            traj = get_debugging_cartesian_trajectory()
        elif acq_type == 'spiral':
            traj = get_spiral_trajectory(volume_size, **acq_kwargs)
        else:
            raise NotImplementedError(f'{acq_type} dataset not implemented yet.')
        if compute_dcomp:
            interpob = nfft_ob._extract_nufft_interpob()
            nufftob_forw = kbnufft_forward(interpob)
            nufftob_back = kbnufft_adjoint(interpob)
            dcomp = calculate_radial_dcomp_tf(
                interpob,
                nufftob_forw,
                nufftob_back,
                traj[0],
            )
        # we add a batch dimension to the trajectory
        traj = traj[None, ...]
        # need to add batch and coil dimension to the volume
        nc_kspace = nufft(nfft_ob, volume[None, None, ...], traj, volume_size)
        nc_kspace_scaled = nc_kspace * scale_factor
        volume_scaled = volume * scale_factor
        volume_channeled = volume_scaled[..., None]
        nc_kspaces_channeled = nc_kspace_scaled[..., None]
        orig_shape = tf.ones([tf.shape(kspaces)[0]], dtype=tf.int32) * tf.shape(kspaces)[-1]
        extra_args = (orig_shape,)
        if compute_dcomp:
            dcomp = tf.ones([1, tf.shape(dcomp)[0]], dtype=dcomp.dtype) * dcomp[None, :]
            extra_args += (dcomp,)
        return (nc_kspaces_channeled, traj, extra_args), volume_channeled
    return tf.function(
        from_volume_to_nc_kspace_and_traj,
        input_signature=[
            tf.TensorSpec([None, None, None], dtype=tf.complex64),
        ],
        autograph=True,
        experimental_relax_shapes=True,
    )
