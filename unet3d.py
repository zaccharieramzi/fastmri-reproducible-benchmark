"""Largely inspired by https://github.com/zhixuhao/unet/blob/master/model.py"""
import os
import tempfile

from keras import activations
from keras import backend as K
from keras.layers import Conv3D, MaxPooling3D, concatenate, Dropout, UpSampling3D, Input, AveragePooling3D, BatchNormalization
from keras.models import Model
from keras.models import load_model
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
        rec_input = pooling(pool_size=(2, 2))(left_u)
        rec_output = unet_rec3d(
            inputs=rec_input,
            kernel_size=kernel_size,
            n_layers=n_layers-1,
            layers_n_channels=layers_n_channels[1:],
            layers_n_non_lins=layers_n_non_lins[1:],
            pool=pool,
            non_relu_contract=non_relu_contract,
        )
        merge = concatenate([left_u, UpSampling3D(size=(2, 2))(rec_output)], axis=3)
        output = chained_convolutions3d(
            merge,
            n_channels=n_channels//2,
            n_non_lins=n_non_lins,
            kernel_size=kernel_size,
        )
    return output


def unet3d(
        with_extra_sigmoid=False,
        pretrained_weights=None,
        input_size=(48, 320, 320, 1),
        kernel_size=3,
        n_layers=1,
        layers_n_channels=1,
        layers_n_non_lins=1,
        non_relu_contract=False,
        pool='max',
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
    new_output = Conv3D(
        input_size[-1],
        kernel_size,
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal',
    )(output)
    model = Model(inputs=inputs, outputs=new_output)
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_absolute_error', metrics=['mean_squared_error'])

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
        conv = BatchNormalization()(conv)
    return conv
