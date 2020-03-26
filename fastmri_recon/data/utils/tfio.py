import tensorflow as tf
import tensorflow_io as tfio


def image_and_kspace_from_h5(fpath):
    spec = {
        '/kspace': tf.TensorSpec(shape=[None, 320, 320], dtype=tf.complex64),
        '/reconstruction_esc': tf.TensorSpec(shape=[None, 640, None], dtype=tf.float32),
    }
    h5_tensors = tfio.IOTensor.from_hdf5(fpath, spec=spec)
    image = h5_tensors('/reconstruction_esc').to_tensor()
    image.set_shape((None, 320, 320))
    kspace = h5_tensors('/kspace').to_tensor()
    kspace.set_shape((None, 640, None))
    return image, kspace
