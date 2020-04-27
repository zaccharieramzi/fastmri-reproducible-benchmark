import tensorflow as tf

from .preprocessing import generic_from_kspace_to_masked_kspace_and_mask
from ..utils.tfio import image_and_kspace_from_h5


# TODO: add datasets for kiki-sep and u-net
# TODO: add test datasets
def train_masked_kspace_dataset_io(path, AF=4, inner_slices=None, rand=False, scale_factor=1):
    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', seed=0)
    image_and_kspace_ds = files_ds.map(
        image_and_kspace_from_h5(inner_slices, rand),
        # TODO: when hdf5 is thread safe, move back to parallel io
        # follow: https://github.com/tensorflow/io/issues/745
        # num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    masked_kspace_ds = image_and_kspace_ds.map(
        generic_from_kspace_to_masked_kspace_and_mask(
            AF=AF,
            scale_factor=scale_factor,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds
