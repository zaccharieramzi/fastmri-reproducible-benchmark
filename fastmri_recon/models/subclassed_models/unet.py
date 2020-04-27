import tensorflow as tf
from tensorflow.keras.models import Model

from ..functional_models.unet import unet


class UnetComplex(Model):
    def __init__(
            self,
            n_output_channels=1,
            kernel_size=3,
            n_layers=1,
            layers_n_channels=1,
            layers_n_non_lins=1,
            res=False,
            **kwargs,
        ):
        super(UnetComplex, self).__init__(**kwargs)
        self.n_output_channels = n_output_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.layers_n_channels = layers_n_channels
        self.layers_n_non_lins = layers_n_non_lins
        self.res = res
        self.unet = unet(
            input_size=(None, None, 2 * self.n_output_channels),  # 2 for real and imag
            kernel_size=self.kernel_size,
            n_layers=self.n_layers,
            layers_n_channels=self.layers_n_channels,
            layers_n_non_lins=self.layers_n_non_lins,
            non_relu_contract=False,
            pool='max',
            compile=False,
        )

    def call(self, inputs):
        outputs = inputs
        outputs = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], axis=-1)
        outputs = self.unet(outputs)
        outputs = tf.complex(
            outputs[..., :self.n_output_channels],
            outputs[..., self.n_output_channels:],
        )
        if self.res:
            outputs = inputs[..., :self.n_output_channels] + outputs
        return outputs
