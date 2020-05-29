import tensorflow as tf

from .preprocessing import generic_from_kspace_to_masked_kspace_and_mask, generic_prepare_mask_and_kspace
from ..utils.h5 import from_train_file_to_image_and_kspace_and_contrast, from_test_file_to_mask_and_kspace_and_contrast
from ..utils.masking.acceleration_factor import tf_af

# TODO: add unet and kikinet datasets
def tf_filename_to_mask_and_kspace_and_contrast(filename):
    def _from_test_file_to_mask_and_kspace_and_contrast_tensor_to_tensor(filename):
        filename_str = filename.numpy()
        mask, kspace, contrast = from_test_file_to_mask_and_kspace_and_contrast(filename_str)
        return tf.convert_to_tensor(mask), tf.convert_to_tensor(kspace), tf.convert_to_tensor(contrast)
    [mask, kspace, contrast] = tf.py_function(
        _from_test_file_to_mask_and_kspace_and_contrast_tensor_to_tensor,
        [filename],
        [tf.bool, tf.complex64, tf.string],
    )
    mask.set_shape((None,))
    kspace.set_shape((None, 640, None))
    return mask, kspace, contrast


def train_masked_kspace_dataset_from_indexable(path, AF=4, inner_slices=None, rand=False, scale_factor=1, contrast=None, n_samples=None):
    selection = [{'inner_slices': inner_slices, 'rand': rand}]
    def _tf_filename_to_image_and_kspace_and_contrast(filename):
        def _from_train_file_to_image_and_kspace_and_contrast_tensor_to_tensor(filename):
            filename_str = filename.numpy()
            image, kspace, contrast = from_train_file_to_image_and_kspace_and_contrast(
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
        kspace_size = n_slices + (640, None)
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
    masked_kspace_ds = image_and_kspace_ds.map(
        generic_from_kspace_to_masked_kspace_and_mask(
            AF=AF,
            scale_factor=scale_factor,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds

def test_masked_kspace_dataset_from_indexable(path, AF=4, scale_factor=1, contrast=None):
    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', seed=0, shuffle=False)
    mask_and_kspace_and_contrast_ds = files_ds.map(
        tf_filename_to_mask_and_kspace_and_contrast,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # contrast filtering
    if contrast:
        mask_and_kspace_and_contrast_ds = mask_and_kspace_and_contrast_ds.filter(
            lambda mask, kspace, tf_contrast: tf_contrast == contrast
        )
    mask_and_kspace_ds = mask_and_kspace_and_contrast_ds.map(
        lambda mask, kspace, tf_contrast: (mask, kspace),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # af filtering
    if AF == 4:
        mask_and_kspace_ds = mask_and_kspace_ds.filter(
            lambda mask, kspace: tf_af(mask) < 5.5
        )
    else:
        mask_and_kspace_ds = mask_and_kspace_ds.filter(
            lambda mask, kspace: tf_af(mask) > 5.5
        )
    masked_kspace_ds = mask_and_kspace_ds.map(
        generic_prepare_mask_and_kspace(
            scale_factor=scale_factor,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds
