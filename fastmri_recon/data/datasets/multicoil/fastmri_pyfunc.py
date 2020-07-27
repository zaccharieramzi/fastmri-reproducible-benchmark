import tensorflow as tf

from .preprocessing import generic_from_kspace_to_masked_kspace_and_mask, generic_prepare_mask_and_kspace
from ...utils.h5 import from_multicoil_train_file_to_image_and_kspace_and_contrast, from_test_file_to_mask_and_kspace_and_contrast_and_image_size, from_test_file_to_mask_and_contrast
from ...utils.masking.acceleration_factor import tf_af

# TODO: add unet and kikinet datasets

def tf_filename_to_mask_and_kspace_and_contrast_and_image_size(filename):
    def _from_test_file_to_mask_and_kspace_and_contrast_and_image_size_tensor_to_tensor(filename):
        filename_str = filename.numpy()
        mask, kspace, contrast, image_size = from_test_file_to_mask_and_kspace_and_contrast_and_image_size(filename_str)
        return tf.convert_to_tensor(mask), tf.convert_to_tensor(kspace), tf.convert_to_tensor(contrast), tf.convert_to_tensor(image_size)
    [mask, kspace, contrast, image_size] = tf.py_function(
        _from_test_file_to_mask_and_kspace_and_contrast_and_image_size_tensor_to_tensor,
        [filename],
        [tf.bool, tf.complex64, tf.string, tf.int32],
    )
    mask.set_shape((None,))
    kspace.set_shape((None, None, None, None))
    image_size.set_shape((None, None))
    return mask, kspace, contrast, image_size

def tf_filename_to_mask_and_contrast_and_filename(filename):
    def _from_test_file_to_mask_and_contrast_tensor_to_tensor(filename):
        filename_str = filename.numpy()
        mask, contrast = from_test_file_to_mask_and_contrast(filename_str)
        return tf.convert_to_tensor(mask), tf.convert_to_tensor(contrast)
    [mask, contrast] = tf.py_function(
        _from_test_file_to_mask_and_contrast_tensor_to_tensor,
        [filename],
        [tf.bool, tf.string],
    )
    mask.set_shape((None,))
    return mask, contrast, filename


def train_masked_kspace_dataset_from_indexable(
        path,
        AF=4,
        inner_slices=None,
        rand=False,
        scale_factor=1,
        contrast=None,
        n_samples=None,
        parallel=True,
        fixed_masks=False,
        output_shape_spec=False,
    ):
    selection = [
        {'inner_slices': inner_slices, 'rand': rand},  # slice selection
        {'rand': parallel, 'keep_dim': False},  # coil selection
    ]
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
        if parallel:
            kspace_size = (None, None)
        else:
            kspace_size = (None, None, None)
        image_size = (None, None)
        image.set_shape(n_slices + image_size)
        kspace.set_shape(n_slices + kspace_size)
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
        num_parallel_calls=tf.data.experimental.AUTOTUNE if rand or parallel else None,
    )
    # contrast filtering
    if contrast:
        image_and_kspace_and_contrast_ds = image_and_kspace_and_contrast_ds.filter(
            lambda image, kspace, tf_contrast: tf_contrast == contrast
        )
    image_and_kspace_ds = image_and_kspace_and_contrast_ds.map(
        lambda image, kspace, tf_contrast: (image, kspace),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if rand or parallel else None,
    )
    if n_samples is not None:
        image_and_kspace_ds = image_and_kspace_ds.take(n_samples)
    masked_kspace_ds = image_and_kspace_ds.map(
        generic_from_kspace_to_masked_kspace_and_mask(
            AF=AF,
            scale_factor=scale_factor,
            parallel=parallel,
            fixed_masks=fixed_masks,
            output_shape_spec=output_shape_spec,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if rand or parallel else None,
    ).repeat()
    if rand or parallel:
        masked_kspace_ds = masked_kspace_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds

def test_masked_kspace_dataset_from_indexable(
        path,
        AF=4,
        scale_factor=1,
        contrast=None,
        n_samples=None,
        output_shape_spec=False,
    ):
    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', seed=0, shuffle=False)
    mask_and_kspace_and_contrast_and_image_size_ds = files_ds.map(
        tf_filename_to_mask_and_kspace_and_contrast_and_image_size,
    )
    # contrast filtering
    if contrast:
        mask_and_kspace_and_contrast_and_image_size_ds = mask_and_kspace_and_contrast_and_image_size_ds.filter(
            lambda mask, kspace, tf_contrast, image_size: tf_contrast == contrast
        )
    mask_and_kspace_and_image_size_ds = mask_and_kspace_and_contrast_and_image_size_ds.map(
        lambda mask, kspace, tf_contrast, image_size: (mask, kspace, image_size),
    )
    # af filtering
    if AF == 4:
        mask_and_kspace_and_image_size_ds = mask_and_kspace_and_image_size_ds.filter(
            lambda mask, kspace, image_size: tf_af(mask) < 5.5
        )
    else:
        mask_and_kspace_and_image_size_ds = mask_and_kspace_and_image_size_ds.filter(
            lambda mask, kspace, image_size: tf_af(mask) > 5.5
        )
    if n_samples is not None:
        mask_and_kspace_and_image_size_ds = mask_and_kspace_and_image_size_ds.take(n_samples)
    masked_kspace_ds = mask_and_kspace_and_image_size_ds.map(
        generic_prepare_mask_and_kspace(
            scale_factor=scale_factor,
            AF=AF,
            output_shape_spec=output_shape_spec,
        )
    )

    return masked_kspace_ds

def test_filenames(path, AF=4, contrast=None, n_samples=None):
    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', seed=0, shuffle=False)
    mask_and_contrast_and_filename_ds = files_ds.map(
        tf_filename_to_mask_and_contrast_and_filename,
    )
    # contrast filtering
    if contrast:
        mask_and_contrast_and_filename_ds = mask_and_contrast_and_filename_ds.filter(
            lambda mask, tf_contrast, filename: tf_contrast == contrast
        )
    mask_and_filename_ds = mask_and_contrast_and_filename_ds.map(
        lambda mask, tf_contrast, filename: (mask, filename),
    )
    # af filtering
    if AF == 4:
        mask_and_filename_ds = mask_and_filename_ds.filter(
            lambda mask, filename: tf_af(mask) < 5.5
        )
    else:
        mask_and_filename_ds = mask_and_filename_ds.filter(
            lambda mask, filename: tf_af(mask) > 5.5
        )
    filename_ds = mask_and_filename_ds.map(
        lambda mask, filename: filename,
    )
    if n_samples is not None:
        filename_ds = filename_ds.take(n_samples)
    return filename_ds
