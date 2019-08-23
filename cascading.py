from keras.layers import Input, Lambda, Conv2D, Add
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

from pdnet_crop import tf_adj_op, tf_op, concatenate_real_imag, complex_from_half, tf_crop
from utils import keras_psnr, keras_ssim


def cascade_net(input_size=(640, None, 1), n_cascade=5, n_convs=5, n_filters=16, lr=1e-3):
    mask_shape = input_size[:-1]
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')
    lamb = 1.0
    m = lamb / (1 + lamb)
    zero_filled = Lambda(tf_adj_op, output_shape=input_size, name='ifft_simple')([kspace_input, mask])


    image = zero_filled
    for i in range(n_cascade):
        # residual convolution
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
        image = Add(name='res_connex_{i}'.format(i=i+1))([image, res_image])
        # data consistency layer
        cnn_fft = Lambda(tf_op, output_shape=input_size, name='fft_masked_{i}'.format(i=i+1))([image, mask])
        cnn_fft = Lambda(lambda x: -x, output_shape=input_size)(cnn_fft)
        data_consistency_fourier = Add(name='data_consist_fft_{i}'.format(i=i+1))([kspace_input, cnn_fft])
        data_consistency_image = Lambda(tf_adj_op, output_shape=input_size, name='ifft_masked_{i}'.format(i=i+1))([data_consistency_fourier, mask])
        data_consistency_image = Lambda(lambda x: m * x, output_shape=input_size)(data_consistency_image)
        image = Add(name='data_consist_enf_{i}'.format(i=i+1))([image, data_consistency_image])

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
