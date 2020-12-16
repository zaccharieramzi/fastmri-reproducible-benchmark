import tensorflow as tf
from tensorflow.keras.models import Model

from fastmri_recon.models.subclassed_models.vnet import Vnet
from fastmri_recon.models.utils.pad_for_pool import pad_for_pool_whole_plane


class PostProcessVnet(Model):
    def __init__(self, recon_model, vnet_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.recon_model = recon_model
        self.recon_model.trainable = False
        self.postproc_model = Vnet(post_processing=True, **vnet_kwargs)

    def call(self, inputs):
        outputs = self.recon_model(inputs)
        orig_shape = tf.shape(outputs)
        outputs, _ = pad_for_pool_whole_plane(outputs, self.postproc_model.n_layers)
        outputs = outputs[None]
        outputs = self.postproc_model(outputs)
        outputs = outputs[0]
        outputs = tf.image.resize_with_crop_or_pad(
            outputs,
            orig_shape[1],
            orig_shape[2],
        )
        return outputs

    def predict_batched(self, inputs, batch_size=1):
        outputs = self.recon_model.predict(inputs, batch_size=batch_size)
        orig_shape = tf.shape(outputs)
        outputs, _ = pad_for_pool_whole_plane(tf.constant(outputs), self.postproc_model.n_layers)
        outputs = outputs[None]
        outputs = self.postproc_model.predict_on_batch(outputs)
        outputs = outputs[0]
        outputs = tf.image.resize_with_crop_or_pad(
            outputs,
            orig_shape[1],
            orig_shape[2],
        )
        return outputs.numpy()
