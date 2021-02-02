import tensorflow as tf
from tensorflow.keras.layers import Layer, UpSampling2D, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model

from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.multi_domain_convolution import MultiDomainConv


class ConvBlockMultiDomain(Layer):
    def __init__(self, n_filters, kernel_size=3, non_linearity='relu', n_non_lins=2, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.non_linearity = non_linearity
        self.n_non_lins = n_non_lins
        self.convs = [
            MultiDomainConv(
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

class UpConv(Layer):
    def __init__(self, n_filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv = Conv2D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation=None,
        )
        self.up = UpSampling2D(size=(2, 2))

    def call(self, inputs):
        outputs = self.up(inputs)
        outputs = self.conv(outputs)
        return outputs


class UnetMultiDomain(Model):
    def __init__(
            self,
            n_outputs=1,
            kernel_size=3,
            layers_n_channels=[64, 128, 256, 512, 1024],
            layers_n_non_lins=2,
            non_linearity='relu',
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.layers_n_channels = layers_n_channels
        self.n_layers = len(self.layers_n_channels)
        self.layers_n_non_lins = layers_n_non_lins
        self.non_linearity = non_linearity
        self.down_convs = [
            ConvBlockMultiDomain(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
                n_non_lins=self.layers_n_non_lins,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.down = MaxPooling2D(pool_size=(2, 2), padding='same')
        self.bottom_conv = ConvBlockMultiDomain(
            n_filters=self.layers_n_channels[-1],
            kernel_size=self.kernel_size,
            non_linearity=self.non_linearity,
            n_non_lins=self.layers_n_non_lins,
        )
        self.up_convs = [
            ConvBlockMultiDomain(
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
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.final_conv = Conv2D(
            filters=self.n_outputs,
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
        return outputs
