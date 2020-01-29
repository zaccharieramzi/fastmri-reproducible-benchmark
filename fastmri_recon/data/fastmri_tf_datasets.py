import glob

import tensorflow as tf
import tensorflow_io as tfio

from .data_utils import from_train_file_to_image_and_kspace, from_file_to_kspace
from ..helpers.nn_mri import tf_unmasked_ifft, _tf_crop
from ..helpers.utils import gen_mask_tf

# TODO: add datasets for kiki-sep and u-net

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

def generic_from_kspace_to_masked_kspace_and_mask(AF=4, inner_slices=None, rand=False, scale_factor=1):
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
    return from_kspace_to_masked_kspace_and_mask

def train_masked_kspace_dataset(path, AF=4, inner_slices=None, rand=False, scale_factor=1):
    masked_kspace_ds = tf.data.Dataset.from_generator(
        image_and_kspace_generator,
        (tf.float32, tf.complex64),
        (tf.TensorShape([None, 320, 320]), tf.TensorShape([None, 640, None])),
        args=(path,),
    ).map(
        generic_from_kspace_to_masked_kspace_and_mask(
            AF=AF,
            inner_slices=inner_slices,
            rand=rand,
            scale_factor=scale_factor,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds

def image_and_kspace_from_h5(fpath):
    spec = {
        '/kspace': tf.TensorSpec(shape=[None, None, None], dtype=tf.complex64),
        '/reconstruction_esc': tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    }
    h5_tensors = tfio.IOTensor.from_hdf5(fpath, spec=spec)
    image = h5_tensors('/reconstruction_esc').to_tensor()
    kspace = h5_tensors('/kspace').to_tensor()
    return image, kspace

def train_masked_kspace_dataset_io(path, AF=4, inner_slices=None, rand=False, scale_factor=1):
    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', seed=0)
    image_and_kspace_ds = files_ds.map(
        image_and_kspace_from_h5,
        # TODO: when hdf5 is thread safe, move back to parallel io
        # follow: https://github.com/tensorflow/io/issues/745
        # num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    masked_kspace_ds = image_and_kspace_ds.map(
        generic_from_kspace_to_masked_kspace_and_mask(
            AF=AF,
            inner_slices=inner_slices,
            rand=rand,
            scale_factor=scale_factor,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds

# zero-filled specific utils
def zero_filled(kspace_mask, images):
    kspaces, _ = kspace_mask
    zero_filled_recon = tf.map_fn(
        tf_unmasked_ifft,
        kspaces,
        dtype=tf.complex64,
        parallel_iterations=35,
        back_prop=False,
        infer_shape=False,
    )
    zero_filled_recon_abs = tf.map_fn(
        tf.math.abs,
        zero_filled_recon,
        dtype=tf.float32,
        parallel_iterations=35,
        back_prop=False,
        infer_shape=False,
    )
    zero_filled_cropped_recon = _tf_crop(zero_filled_recon_abs)
    zero_filled_cropped_recon.set_shape((None, 320, 320, 1))
    return zero_filled_cropped_recon, images

# TODO: have a validation setting to allow for proper inference
def normalize_instance(zero_filled_recon_and_image):
    zero_filled_recon, image = zero_filled_recon_and_image
    mean = tf.reduce_mean(zero_filled_recon)
    stddev = tf.math.reduce_std(zero_filled_recon)
    normalized_zero_filled_recon = (zero_filled_recon - mean) / stddev
    normalized_image = (image - mean) / stddev
    return normalized_zero_filled_recon, normalized_image

def normalize_images(zero_filled_recon, images):
    normalized_zero_filled_and_images = tf.map_fn(
        normalize_instance,
        (zero_filled_recon, images),
        dtype=(tf.float32, tf.float32),
        parallel_iterations=35,
        back_prop=False,
        infer_shape=False,
    )
    normalized_zero_filled_and_images[0].set_shape((None, 320, 320, 1))
    normalized_zero_filled_and_images[1].set_shape((None, 320, 320, 1))
    return normalized_zero_filled_and_images

def train_zero_filled_dataset(path, AF=4, norm=False):
    zero_filled_ds = tf.data.Dataset.from_generator(
        image_and_kspace_generator,
        (tf.float32, tf.complex64),
        (tf.TensorShape([None, 320, 320]), tf.TensorShape([None, 640, None])),
        args=(path,),
    ).map(
        generic_from_kspace_to_masked_kspace_and_mask(
            AF=AF,
            inner_slices=None,
            rand=False,
            scale_factor=1,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).map(
        zero_filled,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if norm:
        zero_filled_ds = zero_filled_ds.map(
            normalize_images,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    zero_filled_ds = zero_filled_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return zero_filled_ds

# kiki-specific utils
def inv_fourier(kspaces_and_masks, kspaces_orig):
    images = tf.map_fn(
        tf_unmasked_ifft,
        kspaces_orig,
        dtype=tf.complex64,
        parallel_iterations=35,
        back_prop=False,
        infer_shape=False,
    )
    return kspaces_and_masks, images

def double_kspace_generator(path):
    filenames = glob.glob(path.decode("utf-8") + '*.h5')
    def _gen():
        for filename in filenames:
            kspace = from_file_to_kspace(filename)
            yield (kspace, kspace)
    return _gen()

def train_masked_kspace_kiki(path, AF=4, inner_slices=None, rand=False, scale_factor=1, space='K'):
    masked_kspace_ds = tf.data.Dataset.from_generator(
        double_kspace_generator,
        (tf.complex64, tf.complex64),
        (tf.TensorShape([None, 640, None]), tf.TensorShape([None, 640, None])),
        args=(path,),
    ).map(
        generic_from_kspace_to_masked_kspace_and_mask(
            AF=AF,
            inner_slices=inner_slices,
            rand=rand,
            scale_factor=scale_factor,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if space == 'I':
        masked_kspace_ds = masked_kspace_ds.map(
            inv_fourier,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    masked_kspace_ds = masked_kspace_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return masked_kspace_ds
