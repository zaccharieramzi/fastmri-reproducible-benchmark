import tensorflow as tf

from ..utils.masking.gen_mask_tf import gen_mask_tf


def generic_from_kspace_to_masked_kspace_and_mask(AF=4, scale_factor=1):
    def from_kspace_to_masked_kspace_and_mask(images, kspaces):
        mask = gen_mask_tf(kspaces, accel_factor=AF)
        kspaces_masked = mask * kspaces
        kspaces_scaled = kspaces_masked * scale_factor
        images_scaled = images * scale_factor
        kspaces_channeled = kspaces_scaled[..., None]
        images_channeled = images_scaled[..., None]
        return (kspaces_channeled, mask), images_channeled
    return from_kspace_to_masked_kspace_and_mask

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
