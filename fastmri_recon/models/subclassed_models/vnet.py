import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv3D, LeakyReLU, PReLU, UpSampling3D, MaxPooling3D, Activation
from tensorflow.keras.models import Model

from ..utils.complex import to_complex
from ..utils.fourier import AdjNFFT


class Conv(Layer):
    def __init__(self, n_filters, kernel_size=3, non_linearity='relu', **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.non_linearity = non_linearity
        self.conv = Conv3D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation=None,
        )
        if self.non_linearity == 'lrelu':
            self.act = LeakyReLU(0.1)
        elif self.non_linearity == 'prelu':
            self.act = PReLU(shared_axes=[1, 2, 3])
        else:
            self.act = Activation(self.non_linearity)

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.act(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'non_linearity': self.non_linearity,
        })
        return config

class ConvBlock(Layer):
    def __init__(self, n_filters, kernel_size=3, non_linearity='relu', n_non_lins=2, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.non_linearity = non_linearity
        self.n_non_lins = n_non_lins
        self.convs = [
            Conv(
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
            ) for _ in range(self.n_non_lins)
        ]

    def call(self, inputs):
        outputs = inputs
        for conv in self.convs:
            outputs = conv(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_non_lins': self.n_non_lins,
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'non_linearity': self.non_linearity,
        })
        return config

class UpConv(Layer):
    def __init__(self, n_filters, kernel_size=3, post_processing=False, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.post_processing = post_processing
        self.conv = Conv3D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation=None,
        )
        self.up = UpSampling3D(size=(1 if self.post_processing else 2, 2, 2))

    def call(self, inputs):
        outputs = self.up(inputs)
        outputs = self.conv(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'post_processing': self.post_processing,
        })
        return config


class Vnet(Model):
    def __init__(
            self,
            n_output_channels=1,
            kernel_size=3,
            layers_n_channels=[4],
            layers_n_non_lins=1,
            non_linearity='relu',
            post_processing=False,
            res=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.n_output_channels = n_output_channels
        self.kernel_size = kernel_size
        self.layers_n_channels = layers_n_channels
        self.n_layers = len(self.layers_n_channels)
        self.layers_n_non_lins = layers_n_non_lins
        self.non_linearity = non_linearity
        self.post_processing = post_processing
        self.res = res
        self.down_convs = [
            ConvBlock(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
                n_non_lins=self.layers_n_non_lins,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.down = MaxPooling3D(
            pool_size=(1 if self.post_processing else 2, 2, 2),
            padding='same',
        )
        self.bottom_conv = ConvBlock(
            n_filters=self.layers_n_channels[-1],
            kernel_size=self.kernel_size,
            non_linearity=self.non_linearity,
            n_non_lins=self.layers_n_non_lins,
        )
        self.up_convs = [
            ConvBlock(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
                n_non_lins=self.layers_n_non_lins,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.ups = [
            UpConv(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                post_processing=self.post_processing,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.final_conv = Conv3D(
            filters=self.n_output_channels,
            kernel_size=1,
            padding='same',
            activation=None,
        )

    def call(self, inputs):
        scales = []
        outputs = inputs
        for conv in self.down_convs:
            outputs = conv(outputs)
            scales.append(outputs)
            outputs = self.down(outputs)
        outputs = self.bottom_conv(outputs)
        for scale, conv, up in zip(scales[::-1], self.up_convs[::-1], self.ups[::-1]):
            outputs = up(outputs)
            outputs = tf.concat([outputs, scale], axis=-1)
            outputs = conv(outputs)
        outputs = self.final_conv(outputs)
        if self.res:
            outputs = outputs + inputs
        return outputs


class VnetComplex(Model):
    def __init__(
            self,
            n_input_channels=1,
            n_output_channels=1,
            kernel_size=3,
            layers_n_channels=[4],
            layers_n_non_lins=1,
            res=False,
            non_linearity='relu',
            dealiasing_nc=False,
            im_size=None,
            dcomp=None,
            grad_traj=False,
            **kwargs,
        ):
        super(VnetComplex, self).__init__(**kwargs)
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.kernel_size = kernel_size
        self.layers_n_channels = layers_n_channels
        self.layers_n_non_lins = layers_n_non_lins
        self.res = res
        self.non_linearity = non_linearity
        self.dealiasing_nc = dealiasing_nc
        if self.dealiasing_nc:
            self.adj_op = AdjNFFT(
                im_size=im_size,
                multicoil=False,
                density_compensation=dcomp,
                grad_traj=grad_traj,
            )
        self.vnet = Vnet(
            n_output_channels=2 * self.n_output_channels,
            kernel_size=self.kernel_size,
            layers_n_channels=self.layers_n_channels,
            layers_n_non_lins=self.layers_n_non_lins,
            non_linearity=self.non_linearity,
        )

    def call(self, inputs):
        if self.dealiasing_nc:
            if len(inputs) == 2:
                original_kspace, mask = inputs
                op_args = ()
            else:
                original_kspace, mask, op_args = inputs
            outputs = self.adj_op([original_kspace, mask, *op_args])
            # we do this to match the residual part.
            inputs = outputs
        else:
            outputs = inputs
        # NOTE: for now no padding in 3d case
        outputs = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], axis=-1)
        outputs = self.vnet(outputs)
        outputs = to_complex(outputs, self.n_output_channels)
        if self.res:
            outputs = inputs[..., :self.n_output_channels] + outputs
        if self.dealiasing_nc:
            outputs = tf.abs(outputs)
        return outputs
