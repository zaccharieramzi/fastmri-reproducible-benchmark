"""Base model for Deep Image Prior types of work
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, Layer, Activation, BatchNormalization
from tensorflow.keras.models import Model

from ..utils.fourier import NFFT
from ..utils.complex import to_complex


class BNConv(Layer):
    """docstring for BNConv."""
    def __init__(self, bn=False, non_lin=True, n_filters=128, **kwargs):
        super(BNConv, self).__init__(**kwargs)
        self.bn = bn
        self.non_lin = non_lin
        self.n_filters = n_filters
        self.conv = Conv2D(self.n_filters, 3, padding='same', activation='linear')
        if self.bn:
            self.bnorm = BatchNormalization(epsilon=1e-5, momentum=0.1)
        if self.non_lin:
            self.act = Activation('relu')

    def call(self, x):
        output = self.conv(x)
        if self.bn:
            output = self.bnorm(output)
        if self.non_lin:
            output = self.act(output)
        return output


class ConvBlock(Layer):
    """docstring for ConvBlock."""
    def __init__(self, n_convs=2, bn=False, non_lin=True, n_filters=128, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.n_convs = n_convs
        self.bn = bn
        self.non_lin = non_lin
        self.n_filters = n_filters
        self.convs = [BNConv(bn, non_lin, n_filters) for _ in range(self.n_convs)]

    def call(self, x):
        output = x
        for conv in self.convs:
            output = conv(output)
        return output


class DIPBase(Model):
    """docstring for DIPBase."""
    def __init__(
            self,
            n_hidden=512,
            n_base=20,
            n_up=4,
            n_filters=128,
            im_size=(640, 400),
            bn=False,
            multicoil=False,
            n_coils=1,
            **kwargs,
        ):
        super(DIPBase, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.n_base = n_base
        self.n_up = n_up
        self.n_filters = n_filters
        self.im_size = im_size
        self.bn = bn
        self.multicoil = multicoil
        self.n_coils = n_coils
        self.denses = [Dense(self.n_hidden, 'relu'), Dense(self.n_base**2)]
        self.ups = [UpSampling2D(size=2, interpolation='nearest') for _ in range(self.n_up)]
        self.convs = [ConvBlock(2, self.bn, True, self.n_filters) for _ in range(self.n_up+1)]
        self.convs.append(Conv2D(2 * self.n_coils, 3, padding='same'))
        self.op = NFFT(im_size=self.im_size, multicoil=self.multicoil)

    def call(self, inputs):
        x, ktraj = inputs
        image = self.generate(x)
        image = tf.image.resize_with_crop_or_pad(image, self.im_size[0], self.im_size[1])
        if self.multicoil:
            # we do not use smaps like in the Darestani paper
            # however, because we still want a kspace per coil
            # we need to use a trick where we make the smaps the image
            # and vice versa
            smaps = tf.ones_like(image[..., 0:1], dtype=image.dtype)
            # at this point image has a shape [slices, h, w, coils]
            # we need to make it [slices, coils, h, w]
            image = tf.transpose(image, [0, 3, 1, 2])
            kspace, _ = self.op([smaps, ktraj, image])
        else:
            kspace, _ = self.op([image, ktraj])
        return kspace

    def generate(self, x, fastmri_format=False, output_shape=(320, 320)):
        output = x
        for dense in self.denses:
            output = dense(output)
        output = tf.reshape(output, [-1, self.n_base, self.n_base, 1])
        for i_up in range(self.n_up+1):
            output = self.convs[i_up](output)
            if i_up < self.n_up:
                output = self.ups[i_up](output)
        output = self.convs[-1](output)
        output = to_complex(output, self.n_coils)
        if fastmri_format:
            output = tf.math.abs(output)
            output = tf.image.resize_with_crop_or_pad(output, *output_shape)
            if self.multicoil:
                output = tf.sqrt(tf.reduce_sum(output**2, axis=-1, keepdims=True))
        return output
