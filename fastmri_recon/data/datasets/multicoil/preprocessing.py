from .coil_selection import selected_coil
from ..slice_selection import selected_slices
from ...utils.masking.gen_mask_tf import gen_mask_tf
from ....models.utils.fourier import tf_unmasked_adj_op


def generic_from_kspace_to_masked_kspace_and_mask(AF=4, inner_slices=None, rand=False, scale_factor=1, parallel=True):
    def from_kspace_to_masked_kspace_and_mask(images, kspaces):
        mask = gen_mask_tf(kspaces, accel_factor=AF, multicoil=True)
        if inner_slices is not None:
            slices = selected_slices(kspaces, inner_slices=inner_slices, rand=rand)
            kspaces_sliced = kspaces[slices[0]:slices[1]]
            mask_sliced = mask[slices[0]:slices[1]]
        else:
            kspaces_sliced = kspaces
            mask_sliced = mask
        if parallel:
            i_coil = selected_coil(kspaces_sliced)
            kspaces_sliced = kspaces_sliced[:, i_coil]
            mask_sliced = mask_sliced[:, i_coil]
        images_sliced = tf_unmasked_adj_op(kspaces_sliced[..., None])
        kspaces_masked = mask * kspaces_sliced
        kspaces_scaled = kspaces_masked * scale_factor
        images_scaled = images_sliced * scale_factor
        kspaces_channeled = kspaces_scaled[..., None]
        images_channeled = images_scaled[..., None]
        return (kspaces_channeled, mask_sliced), images_channeled
    return from_kspace_to_masked_kspace_and_mask

# TODO: adapt to multicoil
# def generic_prepare_mask_and_kspace(scale_factor=1):
#     def prepare(mask, kspaces):
#         shape = tf.shape(kspaces)
#         num_cols = shape[-1]
#         mask_shape = tf.ones_like(shape)
#         # TODO: this could be refactored with gen_mask_tf
#         final_mask_shape = tf.concat([
#             mask_shape[:2],
#             tf.expand_dims(num_cols, axis=0),
#         ], axis=0)
#         final_mask_reshaped = tf.reshape(mask, final_mask_shape)
#         fourier_mask = tf.tile(final_mask_reshaped, [shape[0], shape[1], 1])
#         fourier_mask = tf.dtypes.cast(fourier_mask, 'complex64')
#         kspaces_scaled = kspaces * scale_factor
#         kspaces_channeled = kspaces_scaled[..., None]
#         return kspaces_channeled, fourier_mask
#     return prepare
