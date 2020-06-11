import tensorflow as tf
from tensorflow.keras.models import Model

from ..utils.complex import to_complex
from ..utils.pad_for_pool import pad_for_pool


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
        if self.n_scales > 0:
            outputs, n_pad = pad_for_pool(inputs, self.n_scales)
        outputs = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], axis=-1)
        outputs = self.model(outputs)
        outputs = to_complex(outputs, self.n_output_channels)
        if self.n_scales > 0:
            outputs = tf.cond(
                n_pad == 0,
                lambda: outputs,
                lambda: outputs[:, :, n_pad//2:-n_pad//2],
            )
        if self.res:
            outputs = inputs[..., :self.n_output_channels] + outputs
        return outputs
