from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU
from tensorflow.keras.models import Model


def discriminator_model(im_size=320):
    """Build discriminator architecture.
    from https://github.com/RaphaelMeudec/deblur-gan/blob/master/deblurgan/model.py
    """
    ndf, n_layers, k_size = 64, 3, 3
    inputs = Input(shape=(im_size, im_size, 1))

    x = Conv2D(filters=ndf, kernel_size=k_size, strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=k_size, strides=2, padding='same')(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=ndf*nf_mult, kernel_size=k_size, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=k_size, strides=1, padding='same')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model
