from keras.initializers import Constant
from keras.layers import Input, Lambda, Conv2D, Add, Layer
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

from pdnet_crop import tf_adj_op, tf_op, concatenate_real_imag, complex_from_half, tf_crop
from utils import keras_psnr, keras_ssim

class DataConsistency(Layer):
    def __init__(self, mask, kspace_input, sample_weight_init=1.0, **kwargs):
        self.mask = mask
        self.kspace_input = kspace_input
        self.sample_weight_init = sample_weight_init
        super(DataConsistency, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.sample_weight = self.add_weight(
            name='sample_weight',
            shape=(1,),
            initializer=Constant(self.sample_weight_init),
            trainable=True,
        )
        super(DataConsistency, self).build(input_shape)  # Be sure to call this at the end

    def call(self, image):
        cnn_fft = tf_op([image, self.mask])
        data_consistency_fourier = self.kspace_input - cnn_fft
        data_consistency_image = tf_adj_op([data_consistency_fourier, self.mask])
        data_consistency_image = self.sample_weight_init * data_consistency_image
        image = image + data_consistency_image
        return image

    def compute_output_shape(self, input_shape):
        return input_shape


def cascade_net(input_size=(640, None, 1), n_cascade=5, n_convs=5, n_filters=16, sw_init=1.0, lr=1e-3):
    mask_shape = input_size[:-1]
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')
    lamb = 1.0
    m = lamb / (1 + lamb)
    zero_filled = Lambda(tf_adj_op, output_shape=input_size, name='ifft_simple')([kspace_input, mask])


    image = zero_filled
    dc_layer = DataConsistency(mask=mask, kspace_input=kspace_input, sample_weight_init=sw_init)
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
        image = dc_layer(image)


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
