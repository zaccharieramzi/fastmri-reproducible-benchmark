from keras.layers import Input, Lambda
from keras.models import Model

from ..helpers.keras_utils import default_model_compile
from ..helpers.nn_mri import tf_fastmri_format, tf_unmasked_adj_op, tf_unmasked_op, conv2d_complex, enforce_kspace_data_consistency


def kiki_sep_net(previous_net, multiply_scalar, input_size=(640, None, 1), n_convs=5, n_filters=16, noiseless=True, lr=1e-3, to_add='I', last=False, activation='relu'):
    mask_shape = input_size[:-1]
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')
    if previous_net is None:
        kspace = conv2d_complex(kspace_input, n_filters, n_convs, output_shape=input_size, res=False, activation=activation)
        output = kspace
    else:
        previous_net.trainable = False
        if to_add == 'I':
            kspace = previous_net([kspace_input, mask])
            image = Lambda(tf_unmasked_adj_op, output_shape=input_size)(kspace)
            image = conv2d_complex(image, n_filters, n_convs, output_shape=input_size, res=True, activation=activation)
            output = image
            if last:
                output = tf_fastmri_format(output)
        elif to_add == 'K':
            image = previous_net([kspace_input, mask])
            kspace = Lambda(tf_unmasked_op, output_shape=input_size)(image)
            kspace = enforce_kspace_data_consistency(kspace, kspace_input, mask, input_size, multiply_scalar, noiseless)
            # K-net
            kspace = conv2d_complex(kspace, n_filters, n_convs, output_shape=input_size, res=False, activation=activation)
            output = kspace
    model = Model(inputs=[kspace_input, mask], outputs=output)
    default_model_compile(model, lr)
    return model
