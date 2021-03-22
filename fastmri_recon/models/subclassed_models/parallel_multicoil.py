import tensorflow as tf
from tensorflow.keras.models import Model

from ..utils.fastmri_format import _tf_crop


class ParallelMulticoil(Model):
    def __init__(self, submodel, **kwargs):
        super(ParallelMulticoil, self).__init__(**kwargs)
        self.submodel = submodel

    def call(self, inputs):
        reconstructed_slices = tf.map_fn(
            self.call_single_slice,
            inputs,
            fn_output_signature=tf.float32,
            parallel_iterations=2,
            back_prop=False,
        )
        cropped_reconstruction = _tf_crop(reconstructed_slices)
        return cropped_reconstruction


    def call_single_slice(self, inputs):
        reconstructed_coil_images = self.submodel(inputs)
        # rss reconstruction
        rss_reconstruction = tf.norm(reconstructed_coil_images, axis=0)
        return rss_reconstruction
