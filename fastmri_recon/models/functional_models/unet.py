"""Largely inspired by https://github.com/zhixuhao/unet/blob/master/model.py"""
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    concatenate,
    Dropout,
    UpSampling2D,
    Input,
    AveragePooling2D,
    BatchNormalization,
    Lambda,
    LeakyReLU,
    PReLU,
    Subtract,
)
from tensorflow.keras.models import Model

from ..training.compile import default_model_compile
from ..utils.attention import ChannelAttentionBlock
from ..utils.fastmri_format import tf_fastmri_format
from ..utils.fourier import tf_unmasked_adj_op


def _instantiate_non_linearity(non_linearity):
    if non_linearity == 'lrelu':
        non_linearity_inst = LeakyReLU(0.1)
    elif non_linearity == 'prelu':
        non_linearity_inst = PReLU(shared_axes=[1, 2])
    else:
        non_linearity_inst = non_linearity
    return non_linearity_inst

def unet_rec(
        inputs,
        kernel_size=3,
        n_layers=1,
        layers_n_channels=1,
        layers_n_non_lins=1,
        pool='max',
        non_relu_contract=False,
        non_linearity='relu',
        **channel_attention_kwargs,
    ):
    if n_layers == 1:
        last_conv = chained_convolutions(
            inputs,
            n_channels=layers_n_channels[0],
            n_non_lins=layers_n_non_lins[0],
            kernel_size=kernel_size,
            activation=non_linearity,
        )
        output = last_conv
    else:
        # TODO: refactor the following
        n_non_lins = layers_n_non_lins[0]
        n_channels = layers_n_channels[0]
        if non_relu_contract:
            activation = 'linear'
        else:
            activation = non_linearity
        left_u = chained_convolutions(
            inputs,
            n_channels=n_channels,
            n_non_lins=n_non_lins,
            kernel_size=kernel_size,
            activation=activation,
            **channel_attention_kwargs,
        )
        if pool == 'average':
            pooling = AveragePooling2D
        else:
            pooling = MaxPooling2D
        rec_input = pooling(pool_size=(2, 2))(left_u)
        rec_output = unet_rec(
            inputs=rec_input,
            kernel_size=kernel_size,
            n_layers=n_layers-1,
            layers_n_channels=layers_n_channels[1:],
            layers_n_non_lins=layers_n_non_lins[1:],
            pool=pool,
            non_relu_contract=non_relu_contract,
            non_linearity=non_linearity,
        )
        activation = _instantiate_non_linearity(non_linearity)
        merge = concatenate([
            left_u,
            Conv2D(
                n_channels,
                kernel_size - 1,
                activation=activation,
                padding='same',
                kernel_initializer='glorot_uniform',
            )(UpSampling2D(size=(2, 2))(rec_output))  # up-conv
        ], axis=3)
        output = chained_convolutions(
            merge,
            n_channels=n_channels,
            n_non_lins=n_non_lins,
            kernel_size=kernel_size,
            activation=non_linearity,
            **channel_attention_kwargs,
        )
    return output


def unet(
        pretrained_weights=None,
        input_size=(256, 256, 1),
        n_output_channels=None,
        kernel_size=3,
        n_layers=1,
        layers_n_channels=1,
        layers_n_non_lins=1,
        non_relu_contract=False,
        non_linearity='relu',
        pool='max',
        lr=1e-3,
        compile=True,
        res=False,
        **channel_attention_kwargs,
    ):
    if isinstance(layers_n_channels, int):
        layers_n_channels = [layers_n_channels] * n_layers
    else:
        assert len(layers_n_channels) == n_layers
    if isinstance(layers_n_non_lins, int):
        layers_n_non_lins = [layers_n_non_lins] * n_layers
    else:
        assert len(layers_n_non_lins) == n_layers
    if n_output_channels is None:
        n_output_channels = input_size[-1]
    inputs = Input(input_size)
    output = unet_rec(
        inputs,
        kernel_size=kernel_size,
        n_layers=n_layers,
        layers_n_channels=layers_n_channels,
        layers_n_non_lins=layers_n_non_lins,
        pool=pool,
        non_relu_contract=non_relu_contract,
        non_linearity=non_linearity,
        **channel_attention_kwargs,
    )
    activation = _instantiate_non_linearity(non_linearity)
    output = Conv2D(
        max(4, n_output_channels),
        1,
        # NOTE: this is a breaking change for the results for fastMRI and OASIS
        # we would need to retrain to get the proper results.
        activation=activation,
        padding='same',
        kernel_initializer='glorot_uniform',
    )(output)
    if channel_attention_kwargs:
        output = ChannelAttentionBlock(
            activation=non_linearity,
            **channel_attention_kwargs,
        )(output)
    output = Conv2D(
        n_output_channels,
        1,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
    )(output)
    if res:
        output = Subtract()([inputs, output])
    model = Model(inputs=inputs, outputs=output, name='unet')
    if compile:
        default_model_compile(model, lr)
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def full_unet(
        input_size=(640, None, 1),
        lr=1e-3,
        **unet_kwargs,
    ):
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    zero_filled = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft')(kspace_input)
    image = tf_fastmri_format(zero_filled)
    unet_pred = unet(input_size=(320, 320, 1), compile=False, **unet_kwargs)
    image = unet_pred(image)
    model = Model(inputs=kspace_input, outputs=image)
    default_model_compile(model, lr)

    return model




def old_unet(pretrained_weights=None, input_size=(256, 256, 1), dropout=0.5, kernel_size=3):
    inputs = Input(input_size)
    conv1 = Conv2D(1, kernel_size , activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv5 = Conv2D(1, kernel_size , activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(1, kernel_size , activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv1,up6], axis=3)
    # conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(input_size[-1], kernel_size , activation='sigmoid', padding='same', kernel_initializer='he_normal')(merge6)



    model = Model(input=inputs, output=conv6)

    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def chained_convolutions(
        inputs,
        n_channels=1,
        n_non_lins=1,
        kernel_size=3,
        activation='relu',
        **channel_attention_kwargs,
    ):
    conv = inputs
    for _ in range(n_non_lins):
        activation_inst = _instantiate_non_linearity(activation)
        conv = Conv2D(
            n_channels,
            kernel_size,
            activation=activation_inst,
            padding='same',
            kernel_initializer='glorot_uniform',
        )(conv)
        # conv = BatchNormalization()(conv)
    if channel_attention_kwargs:
        output = ChannelAttentionBlock(
            activation=activation,
            **channel_attention_kwargs,
        )(conv)
    else:
        output = conv
    return output
