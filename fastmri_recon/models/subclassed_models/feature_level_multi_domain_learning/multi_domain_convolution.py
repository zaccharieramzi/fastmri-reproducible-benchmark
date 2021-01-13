import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D

from feature_multidomain_learning.models.complex import to_complex, to_real
from feature_multidomain_learning.models.fourier import ortho_fft2d, ortho_ifft2d

class MultiDomainConv(Layer):
    def __init__(self, n_filters=32, non_linearity=None, kernel_size=1, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.non_linearity = non_linearity
        self.kernel_size = kernel_size
        self.conv_image = Conv2D(
            self.n_filters,
            self.kernel_size,
            padding='same',
            activation=self.non_linearity,
        )
        self.conv_kspace = Conv2D(
            self.n_filters,
            self.kernel_size,
            padding='same',
            activation=self.non_linearity,
        )

    def build(self, input_shape):
        self.n_input_channels = int(input_shape[-1])

    def call(self, image):
        image_complex = to_complex(image, self.n_input_channels//2)
        kspace = ortho_fft2d(image_complex)
        kspace = to_real(kspace)
        out_image = self.conv_image(image)
        out_kspace = self.conv_kspace(kspace)
        out_kspace = to_complex(out_kspace, self.n_filters//2)
        out_kspace = ortho_ifft2d(out_kspace)
        out_kspace = to_real(out_kspace)
        out = tf.concat([
            out_image,
            out_kspace,
        ], axis=-1)
        return out
