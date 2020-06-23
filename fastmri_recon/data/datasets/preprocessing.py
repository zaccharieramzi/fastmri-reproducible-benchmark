import tensorflow as tf

from ..utils.masking.gen_mask_tf import gen_mask_tf
from ..utils.non_cartesian import get_radial_trajectory, get_debugging_cartesian_trajectory
from fastmri_recon.models.utils.fourier import tf_unmasked_adj_op, nufft


def generic_from_kspace_to_masked_kspace_and_mask(AF=4, scale_factor=1, fixed_masks=False):
    def from_kspace_to_masked_kspace_and_mask(images, kspaces):
        mask = gen_mask_tf(kspaces, accel_factor=AF, fixed_masks=fixed_masks)
        kspaces_masked = tf.cast(mask, kspaces.dtype) * kspaces
        kspaces_scaled = kspaces_masked * scale_factor
        images_scaled = images * scale_factor
        kspaces_channeled = kspaces_scaled[..., None]
        images_channeled = images_scaled[..., None]
        return (kspaces_channeled, mask), images_channeled
    return from_kspace_to_masked_kspace_and_mask

def non_cartesian_from_kspace_to_nc_kspace_and_traj(nfft_ob, image_size, acq_type='radial', scale_factor=1, **acq_kwargs):
    def from_kspace_to_nc_kspace_and_traj(images, kspaces):
        if acq_type == 'radial':
            traj = get_radial_trajectory(**acq_kwargs)
        elif acq_type == 'cartesian':
            traj = get_debugging_cartesian_trajectory()
        else:
            raise NotImplementedError(f'{acq_type} dataset not implemented yet.')
        traj = tf.repeat(traj, tf.shape(images)[0], axis=0)
        orig_image = tf_unmasked_adj_op(kspaces[..., None])
        nc_kspace = nufft(nfft_ob, orig_image[:, None, ..., 0], traj, image_size)
        nc_kspace_scaled = nc_kspace * scale_factor
        images_scaled = images * scale_factor
        images_channeled = images_scaled[..., None]
        nc_kspaces_channeled = nc_kspace_scaled[..., None]
        orig_shape = tf.ones([tf.shape(kspaces)[0]], dtype=tf.int32) * tf.shape(kspaces)[-1]
        return (nc_kspaces_channeled, traj, (orig_shape,)), images_channeled
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
