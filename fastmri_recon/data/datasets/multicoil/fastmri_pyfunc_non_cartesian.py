import tensorflow as tf
from tfkbnufft.kbnufft import KbNufftModule

from .preprocessing import non_cartesian_from_kspace_to_nc_kspace_and_traj
from ...utils.h5 import from_multicoil_train_file_to_image_and_kspace_and_contrast


def train_nc_kspace_dataset_from_indexable(
        path,
        image_size,
        inner_slices=None,
        rand=False,
        scale_factor=1,
        contrast=None,
        n_samples=None,
        acq_type='radial',
        compute_dcomp=True,  # for backwards compatibility
        **acq_kwargs,
    ):
    if not compute_dcomp:
        raise NotImplementedError('Non-cartesian multicoil is not implemented without density compensation.')
    selection = [{'inner_slices': inner_slices, 'rand': rand}]
    def _tf_filename_to_image_and_kspace_and_contrast(filename):
        def _from_train_file_to_image_and_kspace_and_contrast_tensor_to_tensor(filename):
            filename_str = filename.numpy()
            image, kspace, contrast = from_multicoil_train_file_to_image_and_kspace_and_contrast(
                filename_str,
                selection=selection,
            )
            return tf.convert_to_tensor(image), tf.convert_to_tensor(kspace), tf.convert_to_tensor(contrast)
        [image, kspace, contrast] = tf.py_function(
            _from_train_file_to_image_and_kspace_and_contrast_tensor_to_tensor,
            [filename],
            [tf.float32, tf.complex64, tf.string],
        )
        if rand:
            n_slices = (1,)
        else:
            n_slices = (inner_slices,)
        kspace_size = n_slices + (None, 640, None)
        image_size = n_slices + (320, 320)
        image.set_shape(image_size)
        kspace.set_shape(kspace_size)
        return image, kspace, contrast

    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', shuffle=False)
    # this makes sure the file selection is happening once when using less than
    # all samples
    files_ds = files_ds.shuffle(
        buffer_size=1000,
        seed=0,
        reshuffle_each_iteration=False,
    )
    image_and_kspace_and_contrast_ds = files_ds.map(
        _tf_filename_to_image_and_kspace_and_contrast,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # contrast filtering
    if contrast:
        image_and_kspace_and_contrast_ds = image_and_kspace_and_contrast_ds.filter(
            lambda image, kspace, tf_contrast: tf_contrast == contrast
        )
    image_and_kspace_ds = image_and_kspace_and_contrast_ds.map(
        lambda image, kspace, tf_contrast: (image, kspace),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if n_samples is not None:
        image_and_kspace_ds = image_and_kspace_ds.take(n_samples)
    nufft_ob = KbNufftModule(
        im_size=image_size,
        grid_size=None,
        norm='ortho',
    )
    masked_kspace_ds = image_and_kspace_ds.map(
        non_cartesian_from_kspace_to_nc_kspace_and_traj(
            nufft_ob,
            image_size,
            acq_type=acq_type,
            scale_factor=scale_factor,
            **acq_kwargs,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if rand or inner_slices is not None else None,
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds
