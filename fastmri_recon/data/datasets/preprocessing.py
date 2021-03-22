import math
import multiprocessing
import numpy as np

from scipy.interpolate import griddata
import tensorflow as tf
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.mri.dcomp_calc import calculate_density_compensator

from fastmri_recon.data.utils.crop import adjust_image_size
from ..utils.masking.gen_mask_tf import gen_mask_tf, gen_mask_equidistant_tf
from ..utils.non_cartesian import get_radial_trajectory, get_debugging_cartesian_trajectory, get_spiral_trajectory
from fastmri_recon.models.utils.fourier import tf_unmasked_adj_op, nufft, FFTBase


def generic_from_kspace_to_masked_kspace_and_mask(
        AF=4,
        scale_factor=1,
        fixed_masks=False,
        mask_type='random',
        batch_size=None,
        target_image_size=(640, 400),
    ):
    def from_kspace_to_masked_kspace_and_mask(images, kspaces):
        if batch_size is not None and batch_size > 1:
            fft = FFTBase(False)
            complex_images = fft.adj_op(kspaces[..., None])[..., 0]
            complex_images_padded = adjust_image_size(
                complex_images,
                target_image_size,
                multicoil=False,
            )
            kspaces = fft.op(complex_images_padded[..., None])[..., 0]
        if mask_type == 'random':
            mask = gen_mask_tf(kspaces, accel_factor=AF, fixed_masks=fixed_masks)
        else:
            mask = gen_mask_equidistant_tf(kspaces, accel_factor=AF)
        kspaces_masked = tf.cast(mask, kspaces.dtype) * kspaces
        kspaces_scaled = kspaces_masked * scale_factor
        images_scaled = images * scale_factor
        kspaces_channeled = kspaces_scaled[..., None]
        images_channeled = images_scaled[..., None]
        if batch_size is not None:
            return (kspaces_channeled[0], mask[0]), images_channeled[0]
        else:
            return (kspaces_channeled, mask), images_channeled
    return from_kspace_to_masked_kspace_and_mask

def grid_non_cartesian(traj, nc_kspace, X_grid, Y_grid):
    grid = griddata(traj.T, nc_kspace[0], (X_grid, Y_grid), fill_value=0)
    return grid.T.astype(np.complex64)

def non_cartesian_from_kspace_to_nc_kspace_and_traj(nfft_ob, image_size, acq_type='radial', scale_factor=1, compute_dcomp=False, gridding=False, **acq_kwargs):
    def from_kspace_to_nc_kspace_and_traj(images, kspaces):
        if acq_type == 'radial':
            traj = get_radial_trajectory(image_size, **acq_kwargs)
        elif acq_type == 'cartesian':
            traj = get_debugging_cartesian_trajectory()
        elif acq_type == 'spiral':
            traj = get_spiral_trajectory(image_size, **acq_kwargs)
        else:
            raise NotImplementedError(f'{acq_type} dataset not implemented yet.')
        if compute_dcomp:
            interpob = nfft_ob._extract_nufft_interpob()
            nufftob_back = kbnufft_adjoint(interpob, multiprocessing=True)
            nufftob_forw = kbnufft_forward(interpob, multiprocessing=True)
            dcomp = calculate_density_compensator(
                interpob,
                nufftob_forw,
                nufftob_back,
                traj[0],
            )
        traj = tf.repeat(traj, tf.shape(images)[0], axis=0)
        orig_image = tf_unmasked_adj_op(kspaces[..., None])
        nc_kspace = nufft(nfft_ob, orig_image[:, None, ..., 0], traj, image_size)
        nc_kspace_scaled = nc_kspace * scale_factor
        images_scaled = images * scale_factor
        # Here implement gridding
        if gridding:
            pi = tf.constant(math.pi)
            def tf_grid_nc(nc_kspace_traj):
                nc_kspace, traj = nc_kspace_traj
                X_grid, Y_grid = tf.meshgrid(
                    tf.range(-pi, pi, 2*pi / image_size[0]),
                    tf.range(-pi, pi, 2*pi / image_size[1]),
                )
                nc_kspace = tf.numpy_function(
                    grid_non_cartesian,
                    [traj, nc_kspace, X_grid, Y_grid],
                    tf.complex64,
                )
                return nc_kspace
            nc_kspace_scaled = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(
                tf_grid_nc,
                (nc_kspace_scaled, traj),
                fn_output_signature=tf.complex64,
                parallel_iterations=multiprocessing.cpu_count(),
            ))
            nc_kspace_scaled.set_shape([
                None,
                *image_size,
            ])
            traj = tf.ones_like(nc_kspace_scaled)
        images_channeled = images_scaled[..., None]
        nc_kspaces_channeled = nc_kspace_scaled[..., None]
        orig_shape = tf.ones([tf.shape(kspaces)[0]], dtype=tf.int32) * tf.shape(kspaces)[-1]
        extra_args = (orig_shape,)
        if compute_dcomp:
            dcomp = tf.ones([tf.shape(kspaces)[0], tf.shape(dcomp)[0]], dtype=dcomp.dtype) * dcomp[None, :]
            extra_args += (dcomp,)
        if gridding:
            return (nc_kspaces_channeled, traj), images_channeled
        else:
            return (nc_kspaces_channeled, traj, extra_args), images_channeled
    return tf.function(
        from_kspace_to_nc_kspace_and_traj,
        input_signature=[
            tf.TensorSpec([None, 320, 320]),
            tf.TensorSpec([None, 640, None], dtype=tf.complex64),
        ],
        autograph=True,
        experimental_relax_shapes=True,
    )

def generic_prepare_mask_and_kspace(scale_factor=1):
    def prepare(mask, kspaces):
        shape = tf.shape(kspaces)
        num_cols = shape[-1]
        mask_shape = tf.ones_like(shape)
        # TODO: this could be refactored with gen_mask_tf
        final_mask_shape = tf.concat([
            mask_shape[:2],
            tf.expand_dims(num_cols, axis=0),
        ], axis=0)
        final_mask_reshaped = tf.reshape(mask, final_mask_shape)
        fourier_mask = tf.tile(final_mask_reshaped, [shape[0], shape[1], 1])
        fourier_mask = tf.dtypes.cast(fourier_mask, 'complex64')
        kspaces_scaled = kspaces * scale_factor
        kspaces_channeled = kspaces_scaled[..., None]
        return kspaces_channeled, fourier_mask
    return prepare

def create_noisy_training_pair_fun(noise_std=30, scale_factor=1):
    def create_noisy_training_pair(images):
        noise = tf.random.normal(tf.shape(images), stddev=noise_std)
        images_scaled = images * scale_factor
        images_noisy = images_scaled + noise
        return images_noisy[..., None], images_scaled[..., None]
    return create_noisy_training_pair
