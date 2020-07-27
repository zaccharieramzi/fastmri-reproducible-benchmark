import tensorflow as tf
from tensorflow.keras.models import Model

from ..utils.complex import to_complex
from ..utils.fastmri_format import tf_fastmri_format
from ..utils.fourier import IFFT
from ..utils.pad_for_pool import pad_for_pool


class MultiscaleComplex(Model):
    def __init__(
            self,
            model,
            res=False,
            n_scales=0,
            n_output_channels=1,
            fastmri_format=False,
            **kwargs,
        ):
        super(MultiscaleComplex, self).__init__(**kwargs)
        self.model = model
        self.res = res
        self.n_scales = n_scales
        self.n_output_channels = n_output_channels
        self.fastmri_format = fastmri_format
        if self.fastmri_format:
            self.adj_op = IFFT(masked=False, multicoil=False)

    def call(self, inputs):
        if not self.fastmri_format:
            outputs = inputs
        else:
            outputs = inputs[0]
            outputs = self.adj_op(outputs)
            # this is to be consistent for residual connexion
            inputs = outputs
        if self.n_scales > 0:
            outputs, padding = pad_for_pool(inputs, self.n_scales)
        outputs = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], axis=-1)
        outputs = self.model(outputs)
        outputs = to_complex(outputs, self.n_output_channels)
        if self.n_scales > 0:
            outputs = tf.cond(
                tf.reduce_sum(padding) == 0,
                lambda: outputs,
                lambda: outputs[:, :, padding[0]:-padding[1]],
            )
        if self.res:
            outputs = inputs[..., :self.n_output_channels] + outputs
        if self.fastmri_format:
            outputs = tf_fastmri_format(outputs)
        return outputs
