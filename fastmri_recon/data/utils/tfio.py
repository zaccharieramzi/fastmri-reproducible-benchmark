import tensorflow as tf
import tensorflow_io as tfio

def image_and_kspace_from_h5(inner_slices=None, rand=False):
    def _image_and_kspace_from_h5(fpath):
        spec = {
            '/kspace': tf.TensorSpec(shape=[None, 320, 320], dtype=tf.complex64),
            '/reconstruction_esc': tf.TensorSpec(shape=[None, 640, None], dtype=tf.float32),
        }
        h5_tensors = tfio.IOTensor.from_hdf5(fpath, spec=spec)
        h5_kspace = h5_tensors('/kspace')
        n_slices = h5_kspace.shape[0]
        if inner_slices:
            slice_start = n_slices // 2 - inner_slices // 2
            slice_end = slice_start + inner_slices
        else:
            slice_start = 0
            slice_end = n_slices
        if rand:
            i_slice = tf.random.uniform(
                shape=(),
                minval=slice_start,
                maxval=slice_end,
                dtype=tf.int64,
            )
            slices = (i_slice, i_slice + 1)
        else:
            slices = (slice_start, slice_end)
        kspace = h5_kspace[slices[0]:slices[1]]
        kspace.set_shape((None, 640, None))
        image = h5_tensors('/reconstruction_esc')[slices[0]:slices[1]]
        image.set_shape((None, 320, 320))
        return image, kspace
    return _image_and_kspace_from_h5
