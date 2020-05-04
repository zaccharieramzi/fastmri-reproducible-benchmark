import tensorflow as tf
from tensorflow.keras.models import Model

from ..functional_models.unet import unet
from ..utils.complex import to_complex


class UnetComplex(Model):
    def __init__(
            self,
            n_input_channels=1,
            n_output_channels=1,
            kernel_size=3,
            n_layers=1,
            layers_n_channels=1,
            layers_n_non_lins=1,
            res=False,
            non_linearity='relu',
            **kwargs,
        ):
        super(UnetComplex, self).__init__(**kwargs)
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.layers_n_channels = layers_n_channels
        self.layers_n_non_lins = layers_n_non_lins
        self.res = res
        self.non_linearity = non_linearity
        self.unet = unet(
            input_size=(None, None, 2 * self.n_input_channels),  # 2 for real and imag
            n_output_channels=2 * self.n_output_channels,
            kernel_size=self.kernel_size,
            n_layers=self.n_layers,
            layers_n_channels=self.layers_n_channels,
            layers_n_non_lins=self.layers_n_non_lins,
            non_relu_contract=False,
            non_linearity=self.non_linearity,
            pool='max',
            compile=False,
        )

    def call(self, inputs):
        outputs = inputs
        n_pad = 2**self.n_layers - tf.math.floormod(tf.shape(inputs)[-2], 2**(self.n_layers-1))
        paddings = [
            (0, 0),
            (0, 0),  # here in the context of fastMRI there is nothing to worry about because the dim is 640 (128 x 5)
            (n_pad//2, n_pad//2),
            (0, 0),
        ]
        outputs = tf.pad(outputs, paddings)
        outputs = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], axis=-1)
        outputs = self.unet(outputs)
        outputs = to_complex(outputs, self.n_output_channels)
        outputs = outputs[:, :, n_pad//2:-n_pad//2]
        if self.res:
            outputs = inputs[..., :self.n_output_channels] + outputs
        return outputs
