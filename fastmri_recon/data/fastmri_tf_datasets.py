import glob

import tensorflow as tf

from .data_utils import from_train_file_to_image_and_kspace
from ..helpers.utils import gen_mask_tf

def image_and_kspace_generator(path):
    filenames = glob.glob(path.decode("utf-8") + '*.h5')
    return (from_train_file_to_image_and_kspace(filename) for filename in filenames)

def selected_slices(kspaces, inner_slices=8, rand=True):
    n_slices = tf.shape(kspaces)[0]
    slice_start = n_slices // 2 - inner_slices // 2
    if rand:
        i_slice = tf.random.uniform(
            shape=(1,),
            minval=slice_start,
            maxval=slice_start + inner_slices - 1,
            dtype=tf.int32,
        )
        slices = (i_slice, i_slice + 1)
    else:
        slices = (slice_start, slice_start + inner_slices)
    return slices

def train_masked_kspace_dataset(path, AF=4, inner_slices=None, rand=False, scale_factor=1):
    def from_kspace_to_masked_kspace_and_mask(images, kspaces):
        mask = gen_mask_tf(kspaces, accel_factor=AF)
        kspaces_masked = mask * kspaces
        if inner_slices is not None:
            slices = selected_slices(kspaces_masked, inner_slices=inner_slices, rand=rand)
            kspaces_sliced = kspaces_masked[slices[0][0]:slices[1][0]]
            images_sliced = images[slices[0][0]:slices[1][0]]
            mask_sliced = mask[slices[0][0]:slices[1][0]]
        else:
            kspaces_sliced = kspaces_masked
            images_sliced = images
            mask_sliced = mask
        kspaces_scaled = kspaces_sliced * scale_factor
        images_scaled = images_sliced * scale_factor
        kspaces_channeled = kspaces_scaled[..., None]
        images_channeled = images_scaled[..., None]
        return (kspaces_channeled, mask_sliced), images_channeled

    masked_kspace_ds = tf.data.Dataset.from_generator(
        image_and_kspace_generator,
        (tf.float32, tf.complex64),
        (tf.TensorShape([None, 320, 320]), tf.TensorShape([None, 640, None])),
        args=(path,),
    ).map(
        from_kspace_to_masked_kspace_and_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds
