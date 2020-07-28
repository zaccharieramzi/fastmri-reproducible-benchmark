import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, Layer
from tensorflow.keras.models import Model
from tf_complex.convolutions import ComplexConv2D


class ConvBlock(Layer):
    def __init__(self, n_convs=2, activation='crelu', n_filters=4, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.n_convs = n_convs
        self.activation = activation
        self.n_filters = n_filters
        self.convs = [
            ComplexConv2D(
                n_filters=self.n_filters,
                kernel_size=3,
                activation=self.activation,
                padding='same',
                kernel_initializer='glorot_uniform',
            )
            for i in range(self.n_convs)
        ]

    def call(self, inputs):
        outputs = inputs
        for conv in self.convs:
            outputs = conv(outputs)
        return outputs

class UpConv2D(Layer):
    def __init__(self, activation='crelu', n_filters=4, **kwargs):
        super(UpConv2D, self).__init__(**kwargs)
        self.activation = activation
        self.n_filters = n_filters
        self.conv = ComplexConv2D(
            n_filters=self.n_filters,
            kernel_size=2,
            activation=self.activation,
            padding='same',
            kernel_initializer='glorot_uniform',
        )
        self.up = UpSampling2D(size=(2, 2))

    def call(self, inputs):
        outputs = self.up(inputs)
        outputs = self.conv(outputs)
        return outputs

class UNet(Model):
    def __init__(
            self,
            n_outputs=1,
            n_scales=1,
            base_n_filters=4,
            n_convs=2,
            activation='crelu',
            res=False,
            **kwargs,
        ):
        super(UNet, self).__init__(**kwargs)
        self.n_outputs = n_outputs
        self.n_scales = n_scales
        self.base_n_filters = base_n_filters
        self.n_convs = n_convs
        self.activation = activation
        self.res = res
        self.downward_conv_blocks = [
            ConvBlock(
                n_convs=self.n_convs,
                activation=self.activation,
                n_filters=self.base_n_filters * 2**i_scale,
            )
            for i_scale in range(self.n_scales-1)
        ]
        self.bottom_conv_block = ConvBlock(
            n_convs=self.n_convs,
            activation=self.activation,
            n_filters=self.base_n_filters * 2**self.n_scales,
        )
        self.upward_conv_blocks = [
            ConvBlock(
                n_convs=self.n_convs,
                activation=self.activation,
                n_filters=self.base_n_filters * 2**i_scale,
            )
            for i_scale in range(self.n_scales-1, -1, -1)
        ]
        self.ups = [
            UpConv2D(
                activation=self.activation,
                n_filters=self.base_n_filters * 2**i_scale,
            )
            for i_scale in range(self.n_scales-1, -1, -1)
        ]
        # we need average pooling here because max pooling for complex
        # is not out of the box
        self.down = AveragePooling2D(pool_size=(2, 2))
        self.before_last_conv = ComplexConv2D(
            n_filters=max(4, 2*self.n_outputs),
            kernel_size=1,
            activation=self.activation,
            padding='same',
            kernel_initializer='glorot_uniform',
        )
        self.last_conv = ComplexConv2D(
            n_filters=2*self.n_outputs,
            kernel_size=1,
            activation='linear',
            padding='same',
            kernel_initializer='glorot_uniform',
        )

    def call(self, inputs):
        downward_path = []
        outputs = inputs
        for conv in self.downward_conv_blocks:
            outputs = conv(outputs)
            downward_path.append(outputs)
            outputs = self.down(outputs)
        outputs = self.bottom_conv_block(outputs)
        for up, conv in zip(self.ups, self.upward_conv_blocks):
            downward_skip = downward_path.pop()
            upward = up(outputs)
            outputs = tf.concat([upward, downward_skip], axis=-1)
            outputs = conv(outputs)
        outputs = self.before_last_conv(outputs)
        outputs = self.last_conv(outputs)
        if self.res:
            outputs = outputs + inputs
        return outputs
