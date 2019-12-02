import glob
import os.path as op

import tensorflow as tf
from tqdm import tqdm

from .data_utils import from_train_file_to_image_and_kspace, from_file_to_kspace
from ..helpers.nn_mri import tf_unmasked_ifft, _tf_crop
from ..helpers.utils import gen_mask_tf

# TODO: add datasets for kiki-sep and u-net

# tf record generation
def filename_image_and_kspace_generator_ds(path):
    filenames = glob.glob(path + '*.h5')
    return ((filename, from_train_file_to_image_and_kspace(filename)) for filename in filenames)

def image_and_kspace_generator(path, file_slice=None):
    filenames = glob.glob(path.decode("utf-8") + '*.h5')
    if file_slice is not None:
        filenames = filenames[file_slice]
    return (from_train_file_to_image_and_kspace(filename) for filename in filenames)

# functions originated from https://www.tensorflow.org/tutorials/load_data/tfrecord#tfexample
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tf_serialize_image_kspace(images, kspaces):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'images': _bytes_feature(tf.io.serialize_tensor(images)),
        'kspaces': _bytes_feature(tf.io.serialize_tensor(kspaces)),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(images, kspaces):
    tf_string = tf.py_function(
        tf_serialize_image_kspace,
        (images, kspaces),  # pass these args to the above function.
        tf.string,
    )      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar

def sliced_ds(path, file_slice=None):
    image_kspace_ds = tf.data.Dataset.from_generator(
        image_and_kspace_generator,
        (tf.float32, tf.complex64),
        (tf.TensorShape([None, 320, 320]), tf.TensorShape([None, 640, None])),
        args=(path, file_slice),
    )
    serialized_ds = image_kspace_ds.map(tf_serialize_example)
    return serialized_ds

def create_tf_records(path, n_samples=973, num_shards=200, wrapper=tqdm):
    n_samples_in_shard = n_samples // num_shards + 1
    file_slices = [slice(i_shard * n_samples_in_shard, (i_shard + 1) * n_samples_in_shard) for i_shard in range(num_shards)]
    for i_record, file_slice in wrapper(enumerate(file_slices)):
        record_filename = f'{path}train-{i_record}.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(record_filename, compression_type='GZIP')
        serialized_ds = sliced_ds(path, file_slice=file_slice)
        writer.write(serialized_ds)


# actual datasets
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

def _parse_image_kspace_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    image_kspace_description = {
        'images': tf.io.FixedLenFeature([], tf.string),
        'kspaces': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, image_kspace_description)

def parse_image_kspace(example_dict):
    images = tf.io.parse_tensor(example_dict['images'], out_type=tf.float32)
    images.set_shape((None, 320, 320))
    kspaces = tf.io.parse_tensor(example_dict['kspaces'], out_type=tf.complex64)
    kspaces.set_shape((None, 640, None))
    return images, kspaces

def train_masked_kspace_dataset(path, AF=4, inner_slices=None, rand=False, scale_factor=1):
    files = tf.data.Dataset.list_files(path + '*.tfrecord')
    raw_image_kspace_ds = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # Create a dictionary describing the features.
    serialized_image_kspace_dataset = raw_image_kspace_ds.map(_parse_image_kspace_function)
    image_kspace_ds = serialized_image_kspace_dataset.map(parse_image_kspace)
    masked_kspace_ds = image_kspace_ds.map(
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
