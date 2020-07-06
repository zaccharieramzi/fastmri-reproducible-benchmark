import tensorflow as tf

from .preprocessing import create_noisy_training_pair_fun
from ..utils.h5 import from_train_file_to_image_and_contrast


def train_noisy_dataset_from_indexable(
        path,
        noise_std=30,
        inner_slices=None,
        rand=False,
        scale_factor=1,
        contrast=None,
        n_samples=None,
    ):
    selection = [{'inner_slices': inner_slices, 'rand': rand}]
    def _tf_filename_to_image_and_contrast(filename):
        def _from_train_file_to_image_and_kspace_and_contrast_tensor_to_tensor(filename):
            filename_str = filename.numpy()
            image, contrast = from_train_file_to_image_and_contrast(
                filename_str,
                selection=selection,
            )
            return tf.convert_to_tensor(image), tf.convert_to_tensor(contrast)
        [image, contrast] = tf.py_function(
            _from_train_file_to_image_and_kspace_and_contrast_tensor_to_tensor,
            [filename],
            [tf.float32, tf.string],
        )
        if rand:
            n_slices = (1,)
        else:
            n_slices = (inner_slices,)
        image_size = n_slices + (320, 320)
        image.set_shape(image_size)
        return image, contrast

    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', shuffle=False)
    # this makes sure the file selection is happening once when using less than
    # all samples
    files_ds = files_ds.shuffle(
        buffer_size=1000,
        seed=0,
        reshuffle_each_iteration=False,
    )
    image_and_contrast_ds = files_ds.map(
        _tf_filename_to_image_and_contrast,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # contrast filtering
    if contrast:
        image_and_contrast_ds = image_and_contrast_ds.filter(
            lambda image, tf_contrast: tf_contrast == contrast
        )
    image_ds = image_and_contrast_ds.map(
        lambda image, tf_contrast: image,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if n_samples is not None:
        image_ds = image_ds.take(n_samples)
    noisy_image_ds = image_ds.map(
        create_noisy_training_pair_fun(
            noise_std=noise_std,
            scale_factor=scale_factor,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return noisy_image_ds
