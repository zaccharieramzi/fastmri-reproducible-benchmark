import tensorflow as tf
from tensorflow.keras.models import Model

from ..utils.complex import to_complex


class MultiscaleComplex(Model):
    def __init__(
            self,
            model,
            res=False,
            n_scales=0,
            n_output_channels=1,
            **kwargs,
        ):
        super(MultiscaleComplex, self).__init__(**kwargs)
        self.model = model
        self.res = res
        self.n_scales = n_scales
        self.n_output_channels = n_output_channels

    def call(self, inputs):
        outputs = inputs
        n_pad = 2**self.n_scales - tf.math.floormod(tf.shape(inputs)[-2], 2**(self.n_scales-1))
        paddings = [
            (0, 0),
            (0, 0),  # here in the context of fastMRI there is nothing to worry about because the dim is 640 (128 x 5)
            (n_pad//2, n_pad//2),
            (0, 0),
        ]
        outputs = tf.pad(outputs, paddings)
        outputs = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], axis=-1)
        outputs = self.model(outputs)
        outputs = to_complex(outputs, self.n_output_channels)
        outputs = outputs[:, :, n_pad//2:-n_pad//2]
        if self.res:
            outputs = inputs[..., :self.n_output_channels] + outputs
        return outputs
