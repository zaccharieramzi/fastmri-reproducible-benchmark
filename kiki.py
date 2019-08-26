from keras.layers import Input, Lambda, Conv2D, Add, Layer
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

from cascading import MultiplyScalar, replace_values_on_mask
from pdnet_crop import tf_adj_op, tf_op, concatenate_real_imag, complex_from_half, tf_crop, tf_unmasked_adj_op, tf_unmasked_op
from utils import keras_psnr, keras_ssim


def mask_tf(x):
    k_data, mask = x
    mask = tf.expand_dims(tf.dtypes.cast(mask, k_data.dtype), axis=-1)
    masked_k_data = tf.math.multiply(mask, k_data)
    return masked_k_data


def kiki_net(input_size=(640, None, 1), n_cascade=5, n_convs=5, n_filters=16, noiseless=True, lr=1e-3):
    mask_shape = input_size[:-1]
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')

    zero_filled = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple')(kspace_input)

    image = zero_filled
    multiply_scalar = MultiplyScalar()
    for i in range(n_cascade):
        # residual convolution (I-net)
        res_image = concatenate_real_imag(image)
        for j in range(n_convs):
            res_image = Conv2D(
                n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='he_normal',
                use_bias=False,
            )(res_image)
        res_image = Conv2D(
            2,
            3,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
        )(res_image)
        res_image = complex_from_half(res_image, 1, input_size)
        image = Add(name='i_res_connex_{i}'.format(i=i+1))([image, res_image])
        # data consistency layer
        cnn_fft = Lambda(tf_unmasked_op, output_shape=input_size, name='fft_simple_{i}'.format(i=i+1))(image)
        if noiseless:
            data_consistency_fourier = Lambda(replace_values_on_mask, output_shape=input_size, name='fft_repl_{i}'.format(i=i+1))([cnn_fft, kspace_input, mask])
        else:
            cnn_fft_masked = Lambda(mask_tf, output_shape=input_size)([cnn_fft, mask])
            cnn_fft_masked = Lambda(lambda x: -x, output_shape=input_size)(cnn_fft_masked)
            data_consistency_fourier = Add(name='data_consist_fft_{i}'.format(i=i+1))([kspace_input, cnn_fft_masked])
            data_consistency_fourier = multiply_scalar(data_consistency_fourier)
            data_consistency_fourier = Add()([data_consistency_fourier, cnn_fft])
        # K-net
        res_k_data = concatenate_real_imag(data_consistency_fourier)
        for j in range(n_convs):
            res_k_data = Conv2D(
                n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='he_normal',
                use_bias=False,
            )(res_k_data)
        res_k_data = Conv2D(
            2,
            3,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
        )(res_k_data)
        res_k_data = complex_from_half(res_k_data, 1, input_size)
        data_consistency_fourier = Add(name='k_res_connex_{i}'.format(i=i+1))([data_consistency_fourier, res_k_data])

        image = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple_{i}'.format(i=i+1))(data_consistency_fourier)

    # module and crop of image
    image = Lambda(tf.math.abs, name='image_module', output_shape=input_size)(image)
    image = Lambda(tf_crop, name='cropping', output_shape=(320, 320, 1))(image)
    model = Model(inputs=[kspace_input, mask], outputs=image)

    model.compile(
        optimizer=Adam(lr=lr),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
    )

    return model
