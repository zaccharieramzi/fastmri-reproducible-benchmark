"""Base model for Deep Image Prior types of work
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D
from tensorflow.keras.models import Model

from ..utils.fourier import NFFT
from ..utils.complex import to_complex


class DIPBase(Model):
    """docstring for DIPBase."""
    def __init__(
            self,
            n_hidden=512,
            n_base=20,
            n_up=4,
            n_filters=128,
            im_size=(640, 400),
            **kwargs,
        ):
        super(DIPBase, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.n_base = n_base
        self.n_up = n_up
        self.n_filters = n_filters
        self.im_size = im_size
        self.denses = [Dense(self.n_hidden, 'relu'), Dense(self.n_base**2)]
        self.ups = [UpSampling2D(size=2, interpolation='nearest') for _ in range(self.n_ups)]
        self.convs = []
        for i_up in range(self.n_up+1):
            conv_1 = Conv2D(self.n_filters, 3, 'same', activation='relu')
            conv_2 = Conv2D(self.n_filters, 3, 'same', activation='relu')
            self.convs += [conv_1, conv_2]
        self.convs.append(Conv2D(2, 3, 'same'))
        self.op = NFFT(im_size=self.im_size)

    def call(self, inputs):
        x, ktraj = inputs
        image = self.generate(x)
        kspace = self.op([image, ktraj])
        return kspace

    def generate(self, x, fastmri_format=False):
        output = x
        for dense in self.denses:
            output = dense(output)
        output = tf.reshape(output, [-1, self.n_base, self.n_base, 1])
        for i_up in range(self.n_up+1):
            output = self.convs[2*i_up](output)
            output = self.convs[2*i_up + 1](output)
            if i_up < self.n_up:
                output = self.ups[i_up](output)
        output = self.convs[-1](output)
        output = to_complex(output, 1)
        if fastmri_format:
            output = tf.math.abs(output)
        return output
