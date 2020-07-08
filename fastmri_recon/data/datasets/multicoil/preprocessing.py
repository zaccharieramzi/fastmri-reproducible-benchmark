import tensorflow as tf
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf

from ...utils.masking.gen_mask_tf import gen_mask_tf
from ...utils.multicoil.smap_extract import extract_smaps, non_cartesian_extract_smaps
from ....models.utils.fourier import tf_unmasked_adj_op, nufft
from ...utils.non_cartesian import get_radial_trajectory, get_debugging_cartesian_trajectory


def generic_from_kspace_to_masked_kspace_and_mask(
        AF=4,
        scale_factor=1,
        parallel=True,
        fixed_masks=False,
    ):
    def from_kspace_to_masked_kspace_and_mask(images, kspaces):
        mask = gen_mask_tf(
            kspaces,
            accel_factor=AF,
            multicoil=not parallel,
            fixed_masks=fixed_masks,
        )
        if parallel:
            images = tf.abs(tf_unmasked_adj_op(kspaces[..., None]))[..., 0]
        else:
            smaps = extract_smaps(kspaces, low_freq_percentage=32//AF)
        kspaces_masked = tf.cast(mask, kspaces.dtype) * kspaces
        kspaces_scaled = kspaces_masked * scale_factor
        images_scaled = images * scale_factor
        kspaces_channeled = kspaces_scaled[..., None]
        images_channeled = images_scaled[..., None]
        if parallel:
            return (kspaces_channeled, mask), images_channeled
        else:
            return (kspaces_channeled, mask, smaps), images_channeled
    return from_kspace_to_masked_kspace_and_mask


def non_cartesian_from_kspace_to_nc_kspace_and_traj(
        nfft_ob,
        image_size,
        acq_type='radial',
        scale_factor=1,
        compute_dcomp=False,
        parallel=False,
        **acq_kwargs
):
    def from_kspace_to_nc_kspace_and_traj(images, kspaces):
        if acq_type == 'radial':
            traj = get_radial_trajectory(image_size, **acq_kwargs)
        elif acq_type == 'cartesian':
            traj = get_debugging_cartesian_trajectory()
        else:
            raise NotImplementedError(f'{acq_type} dataset not implemented yet.')
        interpob = nfft_ob._extract_nufft_interpob()
        nufftob_forw = kbnufft_forward(interpob)
        nufftob_back = kbnufft_adjoint(interpob)
        dcomp = calculate_radial_dcomp_tf(
            interpob,
            nufftob_forw,
            nufftob_back,
            traj[0],
        )
        traj = tf.repeat(traj, tf.shape(images)[0], axis=0)
        orig_image_channels = tf_unmasked_adj_op(kspaces[..., None])[..., 0]
        nc_kspace = nufft(nfft_ob, orig_image_channels, traj, image_size)
        nc_kspaces_channeled = nc_kspace * scale_factor
        images_channeled = images * scale_factor
        orig_shape = tf.ones([tf.shape(kspaces)[0]], dtype=tf.int32) * tf.shape(kspaces)[-1]
        extra_args = (orig_shape,)
        dcomp = tf.ones([tf.shape(kspaces)[0], tf.shape(dcomp)[0]], dtype=dcomp.dtype) * dcomp[None, :]
        if compute_dcomp:
            extra_args += (dcomp,)
        if parallel:
            return (nc_kspaces_channeled, traj, extra_args), images_channeled
        else:
            smaps = non_cartesian_extract_smaps(nc_kspace, traj, dcomp, nufftob_back)
            return (nc_kspaces_channeled, traj, extra_args, smaps), images_channeled
    return tf.function(
        from_kspace_to_nc_kspace_and_traj,
        input_signature=[
            tf.TensorSpec([None, 320, 320]),
            tf.TensorSpec([None, None, 640, None], dtype=tf.complex64),
        ],
        autograph=True,
        experimental_relax_shapes=True,
    )


def generic_prepare_mask_and_kspace(scale_factor=1, AF=4):
    def prepare(mask, kspaces):
        shape = tf.shape(kspaces)
        mask_expanded = mask[None, None, None, :]
        fourier_mask = tf.tile(mask_expanded, [shape[0], 1, 1, 1])
        fourier_mask = tf.dtypes.cast(fourier_mask, tf.uint8)
        smaps = extract_smaps(kspaces, low_freq_percentage=32//AF)
        kspaces_scaled = kspaces * scale_factor
        kspaces_channeled = kspaces_scaled[..., None]
        return kspaces_channeled, fourier_mask, smaps
    return prepare
