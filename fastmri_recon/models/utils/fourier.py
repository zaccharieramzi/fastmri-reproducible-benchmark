import tensorflow as tf
from tensorflow.keras.layers import  Layer
from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, ifftshift, fftshift
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule

from .masking import _mask_tf


def tf_adj_op(y, idx=0):
    x, mask = y
    x_masked = _mask_tf((x, mask))
    x_inv = tf_unmasked_adj_op(x_masked, idx=idx)
    return x_inv

def tf_unmasked_adj_op(x, idx=0):
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.dtypes.cast(tf.math.reduce_prod(tf.shape(x)[1:3]), 'float32')), x.dtype)
    return scaling_norm * tf.expand_dims(fftshift(ifft2d(ifftshift(x[..., idx], axes=[1, 2])), axes=[1, 2]), axis=-1)

def tf_op(y, idx=0):
    x, mask = y
    x_fourier = tf_unmasked_op(x, idx=idx)
    x_masked = _mask_tf((x_fourier, mask))
    return x_masked

def tf_unmasked_op(x, idx=0):
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.dtypes.cast(tf.math.reduce_prod(tf.shape(x)[1:3]), 'float32')), x.dtype)
    return tf.expand_dims(ifftshift(fft2d(fftshift(x[..., idx], axes=[1, 2])), axes=[1, 2]), axis=-1) / scaling_norm

def _compute_scaling_norm(x):
    image_area = tf.reduce_prod(tf.shape(x)[-2:])
    image_area = tf.cast(image_area, 'float32')
    scaling_norm = tf.sqrt(image_area)
    scaling_norm = tf.cast(scaling_norm, x.dtype)
    return scaling_norm

class FFTBase(Layer):
    def __init__(self, masked, multicoil=False, **kwargs):
        super(FFTBase, self).__init__(**kwargs)
        self.masked = masked
        self.multicoil = multicoil
        if self.multicoil:
            self.shift_axes = [2, 3]
        else:
            self.shift_axes = [1, 2]

    def get_config(self):
        config = super(FFTBase, self).get_config()
        config.update({'masked': self.masked})
        config.update({'multicoil': self.multicoil})
        return config

    def op(self, inputs):
        if self.multicoil:
            if self.masked:
                image, mask, smaps = inputs
            else:
                image, smaps = inputs
        else:
            if self.masked:
                image, mask = inputs
            else:
                image = inputs
        image = image[..., 0]
        scaling_norm = _compute_scaling_norm(image)
        if self.multicoil:
            image = tf.expand_dims(image, axis=1)
            image = image * smaps
        shifted_image = fftshift(image, axes=self.shift_axes)
        kspace_shifted = fft2d(shifted_image)
        kspace_unnormed = ifftshift(kspace_shifted, axes=self.shift_axes)
        kspace = kspace_unnormed[..., None] / scaling_norm
        if self.masked:
            kspace = _mask_tf([kspace, mask])
        return kspace

    def adj_op(self, inputs):
        if self.masked:
            if self.multicoil:
                kspace, mask, smaps = inputs
            else:
                kspace, mask = inputs
            kspace = _mask_tf([kspace, mask])
        else:
            if self.multicoil:
                kspace, smaps = inputs
            else:
                kspace = inputs
        kspace = kspace[..., 0]
        scaling_norm = _compute_scaling_norm(kspace)
        shifted_kspace = ifftshift(kspace, axes=self.shift_axes)
        image_shifted = ifft2d(shifted_kspace)
        image_unnormed = fftshift(image_shifted, axes=self.shift_axes)
        image = image_unnormed * scaling_norm
        if self.multicoil:
            image = tf.reduce_sum(image * tf.math.conj(smaps), axis=1)
        image = image[..., None]
        return image

class FFT(FFTBase):
    def call(self, inputs):
        return self.op(inputs)

class IFFT(FFTBase):
    def call(self, inputs):
        return self.adj_op(inputs)

def _pad_for_nufft(image, im_size):
    shape = tf.shape(image)[-1]
    to_pad = im_size[-1] - shape
    padded_image = tf.pad(
        image,
        [
            (0, 0),
            (0, 0),
            (0, 0),
            (to_pad//2, to_pad//2)
        ]
    )
    return padded_image

def _crop_for_pad(image, shape, im_size):
    to_pad = im_size[-1] - shape
    cropped_image = image[..., to_pad//2:-to_pad//2]
    return cropped_image

def _crop_for_nufft(image, im_size):
    shape = tf.shape(image)[-1]
    to_crop = shape - im_size[-1]
    cropped_image = image[..., to_crop//2:-to_crop//2]
    return cropped_image

def nufft(nufft_ob, image, ktraj, image_size=None):
    forward_op = kbnufft_forward(nufft_ob._extract_nufft_interpob())
    shape = tf.shape(image)[-1]
    if image_size is not None:
        image_adapted = tf.cond(
            tf.math.greater(shape, image_size[-1]),
            lambda: _crop_for_nufft(image, image_size),
            lambda: _pad_for_nufft(image, image_size),
        )
    else:
        image_adapted = image
    kspace = forward_op(image_adapted, ktraj)
    return kspace


class NFFTBase(Layer):
    def __init__(self, multicoil=False, im_size=(640, 472), density_compensation=False, **kwargs):
        super(NFFTBase, self).__init__(**kwargs)
        self.multicoil = multicoil
        self.im_size = im_size
        self.nufft_ob = KbNufftModule(
            im_size=im_size,
            grid_size=None,
            norm='ortho',
        )
        self.density_compensation = density_compensation
        self.forward_op = kbnufft_forward(self.nufft_ob._extract_nufft_interpob())
        self.backward_op = kbnufft_adjoint(self.nufft_ob._extract_nufft_interpob())

    def pad_for_nufft(self, image):
        return _pad_for_nufft(image, self.im_size)

    def crop_for_pad(self, image, shape):
        return _crop_for_pad(image, shape, self.im_size)

    def crop_for_nufft(self, image):
        return _crop_for_nufft(image, self.im_size)

    def op(self, inputs):
        if self.multicoil:
            image, ktraj, smaps = inputs
            image = tf.expand_dims(image, axis=1)
            image = image[..., 0] * smaps
        else:
            image, ktraj = inputs
            # for tfkbnufft we need a coil dimension even if there is none
            image = image[:, None, ..., 0]

        kspace = nufft(self.nufft_ob, image, ktraj, image_size=self.im_size)
        shape = tf.ones([tf.shape(image)[0]], dtype=tf.int32) * tf.shape(image)[-1]
        return kspace[..., None], [shape]

    def adj_op(self, inputs):
        if self.multicoil:
            if self.density_compensation:
                kspace, ktraj, smaps, shape, dcomp, = inputs
            else:
                kspace, ktraj, smaps, shape = inputs
        else:
            if self.density_compensation:
                kspace, ktraj, shape, dcomp = inputs
            else:
                kspace, ktraj, shape = inputs
        shape = tf.reshape(shape[0], [])
        if self.density_compensation:
            kspace = tf.cast(dcomp, kspace.dtype) * kspace[..., 0]
        else:
            kspace = kspace[..., 0]
        image = self.backward_op(kspace, ktraj)
        image_reshaped = tf.cond(
            tf.math.greater_equal(shape, self.im_size[-1]),
            lambda: image,
            lambda: self.crop_for_pad(image, shape),
        )
        if self.multicoil:
            image = tf.reduce_sum(image_reshaped * tf.math.conj(smaps), axis=1)
        else:
            image = image_reshaped[:, 0]
        image = image[..., None]
        return image

class NFFT(NFFTBase):
    def call(self, inputs):
        return self.op(inputs)

    def compute_output_shape(self, input_shapes):
        im_shape = input_shapes[0]
        ktraj_shape = input_shapes[1]
        return (im_shape[0], 1, ktraj_shape[-1])

class AdjNFFT(NFFTBase):
    def call(self, inputs):
        return self.adj_op(inputs)

    def compute_output_shape(self, input_shapes):
        kshape = input_shapes[0]
        return (kshape[0], None, None, 1)
