import tensorflow as tf
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.mri.dcomp_calc import calculate_density_compensator

from fastmri_recon.data.utils.crop import adjust_image_size
from fastmri_recon.data.utils.fourier import tf_ortho_ifft2d
from ...utils.masking.gen_mask_tf import gen_mask_tf, gen_mask_equidistant_tf
from ...utils.multicoil.smap_extract import extract_smaps, non_cartesian_extract_smaps
from ....models.utils.fourier import tf_unmasked_adj_op, tf_unmasked_adj_op, nufft, FFTBase
from ...utils.non_cartesian import get_radial_trajectory, get_debugging_cartesian_trajectory, get_spiral_trajectory


def generic_from_kspace_to_masked_kspace_and_mask(
        AF=4,
        scale_factor=1,
        parallel=True,
        fixed_masks=False,
        output_shape_spec=False,
        mask_type='random',
        batch_size=None,
        target_image_size=(640, 400),
    ):
    @tf.function
    def from_kspace_to_masked_kspace_and_mask(images, kspaces):
        if batch_size is not None and batch_size > 1:
            fft = FFTBase(False, multicoil=True, use_smaps=False)
            complex_images = fft.adj_op([kspaces[..., None], None])[..., 0]
            complex_images_padded = adjust_image_size(
                complex_images,
                target_image_size,
                multicoil=True,
            )
            kspaces = fft.op([complex_images_padded[..., None], None])[..., 0]
        if mask_type == 'random':
            mask = gen_mask_tf(
                kspaces,
                accel_factor=AF,
                multicoil=not parallel,
                fixed_masks=fixed_masks,
            )
        elif mask_type == 'equidistant':
            mask = gen_mask_equidistant_tf(
                kspaces,
                accel_factor=AF,
                multicoil=not parallel,
            )
        elif mask_type == 'equidistant_fake':
            mask = gen_mask_equidistant_tf(
                kspaces,
                accel_factor=AF,
                multicoil=not parallel,
                mask_type='fake',
            )
        else:
            raise NotImplementedError(f'{mask_type} masks are not implemented.')
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
            model_inputs = (kspaces_channeled, mask)
        else:
            model_inputs = (kspaces_channeled, mask, smaps)
        if output_shape_spec:
            output_shape = tf.shape(images)[1:][None, :]
            output_shape = tf.tile(output_shape, [tf.shape(images)[0], 1])
            model_inputs += (output_shape,)
        if batch_size is not None:
            images_channeled = images_channeled[0]
            model_inputs = tuple(mi[0] for mi in model_inputs)
        return model_inputs, images_channeled
    return from_kspace_to_masked_kspace_and_mask


def non_cartesian_from_kspace_to_nc_kspace_and_traj(
        nfft_ob,
        image_size,
        acq_type='radial',
        scale_factor=1,
        multiprocessing=True,
        **acq_kwargs
):
    def from_kspace_to_nc_kspace_and_traj(images, kspaces):
        if acq_type == 'radial':
            traj = get_radial_trajectory(image_size, **acq_kwargs)
        elif acq_type == 'cartesian':
            traj = get_debugging_cartesian_trajectory()
        elif acq_type == 'spiral':
            traj = get_spiral_trajectory(image_size, **acq_kwargs)
        else:
            raise NotImplementedError(f'{acq_type} dataset not implemented yet.')
        interpob = nfft_ob._extract_nufft_interpob()
        nufftob_back = kbnufft_adjoint(interpob, multiprocessing=multiprocessing)
        nufftob_forw = kbnufft_forward(interpob, multiprocessing=multiprocessing)
        dcomp = calculate_density_compensator(
            interpob,
            nufftob_forw,
            nufftob_back,
            traj[0],
        )
        traj = tf.repeat(traj, tf.shape(images)[0], axis=0)
        orig_image_channels = tf_ortho_ifft2d(kspaces)
        nc_kspace = nufft(nfft_ob, orig_image_channels, traj, image_size, multiprocessing=multiprocessing)
        nc_kspace_scaled = nc_kspace * scale_factor
        images_scaled = images * scale_factor
        images_channeled = images_scaled[..., None]
        nc_kspaces_channeled = nc_kspace_scaled[..., None]
        orig_shape = tf.ones([tf.shape(kspaces)[0]], dtype=tf.int32) * tf.shape(kspaces)[-1]
        dcomp = tf.ones([tf.shape(kspaces)[0], tf.shape(dcomp)[0]], dtype=dcomp.dtype) * dcomp[None, :]
        extra_args = (orig_shape, dcomp)
        smaps = non_cartesian_extract_smaps(nc_kspace, traj, dcomp, nufftob_back, orig_shape)
        return (nc_kspaces_channeled, traj, smaps, extra_args), images_channeled
    return tf.function(
        from_kspace_to_nc_kspace_and_traj,
        input_signature=[
            tf.TensorSpec([None, 320, 320]),
            tf.TensorSpec([None, None, 640, None], dtype=tf.complex64),
        ],
        autograph=True,
        experimental_relax_shapes=True,
    )


def generic_prepare_mask_and_kspace(scale_factor=1, AF=4, output_shape_spec=False):
    def prepare(mask, kspaces, image_size):
        shape = tf.shape(kspaces)
        mask_expanded = mask[None, None, None, :]
        fourier_mask = tf.tile(mask_expanded, [shape[0], 1, 1, 1])
        fourier_mask = tf.dtypes.cast(fourier_mask, tf.uint8)
        smaps = extract_smaps(kspaces, low_freq_percentage=32//AF)
        kspaces_scaled = kspaces * scale_factor
        kspaces_channeled = kspaces_scaled[..., None]
        if output_shape_spec:
            image_size = image_size[None, :]
            image_size = tf.tile(image_size, [shape[0], 1])
            return kspaces_channeled, fourier_mask, smaps, image_size
        else:
            return kspaces_channeled, fourier_mask, smaps
    return prepare
