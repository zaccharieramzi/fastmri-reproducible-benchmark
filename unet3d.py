"""Largely inspired by https://github.com/zhixuhao/unet/blob/master/model.py"""
from keras.layers import Conv3D, MaxPooling3D, concatenate, UpSampling3D, Input, AveragePooling3D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from utils import keras_psnr, keras_ssim


def unet_rec3d(
        inputs,
        kernel_size=3,
        n_layers=1,
        layers_n_channels=1,
        layers_n_non_lins=1,
        pool='max',
        non_relu_contract=False,
    ):
    if n_layers == 1:
        last_conv = chained_convolutions3d(
            inputs,
            n_channels=layers_n_channels[0],
            n_non_lins=layers_n_non_lins[0],
            kernel_size=kernel_size,
        )
        output = last_conv
    else:
        # TODO: refactor the following
        n_non_lins = layers_n_non_lins[0]
        n_channels = layers_n_channels[0]
        if non_relu_contract:
            activation = 'linear'
        else:
            activation = 'relu'
        left_u = chained_convolutions3d(
            inputs,
            n_channels=n_channels,
            n_non_lins=n_non_lins,
            kernel_size=kernel_size,
            activation=activation,
        )
        if pool == 'average':
            pooling = AveragePooling3D
        else:
            pooling = MaxPooling3D
        rec_input = pooling(pool_size=(1, 2, 2))(left_u)
        rec_output = unet_rec3d(
            inputs=rec_input,
            kernel_size=kernel_size,
            n_layers=n_layers-1,
            layers_n_channels=layers_n_channels[1:],
            layers_n_non_lins=layers_n_non_lins[1:],
            pool=pool,
            non_relu_contract=non_relu_contract,
        )
        merge = concatenate([
            left_u,
            Conv3D(
                n_channels,
                kernel_size - 1,
                activation='relu',
                padding='same',
                kernel_initializer='he_normal',
            )(UpSampling3D(size=(1, 2, 2))(rec_output))  # up-conv
        ], axis=-1)
        output = chained_convolutions3d(
            merge,
            n_channels=n_channels,
            n_non_lins=n_non_lins,
            kernel_size=kernel_size,
        )
    return output


def unet3d(
        pretrained_weights=None,
        input_size=(256, 256, 256, 1),
        kernel_size=3,
        n_layers=1,
        layers_n_channels=1,
        layers_n_non_lins=1,
        non_relu_contract=False,
        pool='max',
        lr=1e-3,
    ):
    if isinstance(layers_n_channels, int):
        layers_n_channels = [layers_n_channels] * n_layers
    else:
        assert len(layers_n_channels) == n_layers
    if isinstance(layers_n_non_lins, int):
        layers_n_non_lins = [layers_n_non_lins] * n_layers
    else:
        assert len(layers_n_non_lins) == n_layers
    inputs = Input(input_size)
    output = unet_rec3d(
        inputs,
        kernel_size=kernel_size,
        n_layers=n_layers,
        layers_n_channels=layers_n_channels,
        layers_n_non_lins=layers_n_non_lins,
        pool=pool,
        non_relu_contract=non_relu_contract,
    )
    output = Conv3D(
        4,
        1,
        activation='linear',
        padding='same',
        kernel_initializer='he_normal',
    )(output)
    output = Conv3D(
        input_size[-1],
        1,
        activation='linear',
        padding='same',
        kernel_initializer='he_normal',
    )(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=lr), loss='mean_absolute_error', metrics=['mean_squared_error', keras_psnr, keras_ssim])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def chained_convolutions3d(inputs, n_channels=1, n_non_lins=1, kernel_size=3, activation='relu'):
    conv = inputs
    for _ in range(n_non_lins):
        conv = Conv3D(
            n_channels,
            kernel_size,
            activation=activation,
            padding='same',
            kernel_initializer='he_normal',
        )(conv)
    return conv
