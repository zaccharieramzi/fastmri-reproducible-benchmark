import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

from ..utils.complex import to_complex


class CNNComplex(Model):
    def __init__(
            self,
            n_convs=3,
            n_filters=16,
            n_output_channels=1,
            activation='relu',
            res=True,
            multicoil=False,
            **kwargs,
        ):
        super(CNNComplex, self).__init__(**kwargs)
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.n_output_channels = n_output_channels
        self.activation = activation
        self.res = res
        self.multicoil = multicoil
        # TODO: maybe have a way to specify non linearity
        self.convs = [
            Conv2D(
                self.n_filters,
                3,
                padding='same',
                activation=self.activation,
                kernel_initializer='glorot_uniform',
            )
            for i in range(self.n_convs-1)
        ]
        self.convs.append(Conv2D(
            2 * self.n_output_channels,
            3,
            padding='same',
            activation='linear',
            kernel_initializer='glorot_uniform',
        ))

    def call(self, inputs):
        outputs = inputs
        if self.multicoil:
            kspace_shape = tf.shape(outputs)
            batch_size = kspace_shape[0]
            n_coils = kspace_shape[1]
            outputs = tf.reshape(
                outputs,
                [batch_size * n_coils, kspace_shape[2], kspace_shape[3], inputs.shape[-1]],
            )
        outputs = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], axis=-1)
        for conv in self.convs:
            outputs = conv(outputs)
        outputs = to_complex(outputs, self.n_output_channels)
        if self.multicoil:
            outputs = tf.reshape(
                outputs,
                [batch_size, n_coils,  kspace_shape[2], kspace_shape[3], self.n_output_channels],
            )
        if self.res:
            outputs = inputs[..., :self.n_output_channels] + outputs
        return outputs
