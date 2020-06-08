import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Layer,
    BatchNormalization,
    Activation,
)


DEFAULT_N_FILTERS_PER_SCALE = [128, 256, 512]
DEFAULT_N_CONVS_PER_SCALE = [3] * 3
DEFAULT_N_FILTERS_PER_SCALE_CONF = [160, 256, 256]
DEFAULT_N_CONVS_PER_SCALE_CONF = [4] * 3


class MWCNNConvBlock(Layer):
    def __init__(self, n_filters=256, kernel_size=3, bn=False, **kwargs):
        super(MWCNNConvBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv = Conv2D(
            self.n_filters,
            self.kernel_size,
            padding='same',
            use_bias=False,
        )
        if bn:
            self.bn = BatchNormalization()
        else:
            self.bn = None
        self.activation = Activation('relu')

    def call(self, inputs):
        outputs = self.conv(inputs)
        if self.bn is not None:
            outputs = self.bn(outputs)
        outputs = self.activation(outputs)
        return outputs

class DWT(Layer):
    def call(self, inputs):
        # taken from
        # https://github.com/lpj-github-io/MWCNNv2/blob/master/MWCNN_code/model/common.py#L65
        x01 = inputs[:, 0::2] / 2
        x02 = inputs[:, 1::2] / 2
        x1 = x01[:, :, 0::2]
        x2 = x02[:, :, 0::2]
        x3 = x01[:, :, 1::2]
        x4 = x02[:, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return tf.concat((x_LL, x_HL, x_LH, x_HH), axis=-1)

class IWT(Layer):
    def call(self, inputs):
        # NOTE: it is for now impossible to do slice assignment in tensorflow
        # there are some on-going GH issues or SO questions but for now
        # tensor_scatter_nd_add seems to be the only way to go.
        # https://stackoverflow.com/questions/62092147/how-to-efficiently-assign-to-a-slice-of-a-tensor-in-tensorflow
        in_shape = tf.shape(inputs)
        batch_size = in_shape[0]
        height = in_shape[1]
        width = in_shape[2]
        # the number of channels can't be unknown for the convolutions
        n_channels = inputs.shape[3] // 4
        outputs = tf.zeros([batch_size, 2 * height, 2 * width, n_channels])
        # for now we only consider greyscale
        x1 = inputs[..., 0:n_channels] / 2
        x2 = inputs[..., n_channels:2*n_channels] / 2
        x3 = inputs[..., 2*n_channels:3*n_channels] / 2
        x4 = inputs[..., 3*n_channels:4*n_channels] / 2
        # in the following, E denotes even and O denotes odd
        x_EE = x1 - x2 - x3 + x4
        x_OE = x1 - x2 + x3 - x4
        x_EO = x1 + x2 - x3 - x4
        x_OO = x1 + x2 + x3 + x4

        # now the preparation to tensor_scatter_nd_add
        height_range_E = 2 * tf.range(height)
        height_range_O = height_range_E + 1
        width_range_E = 2 * tf.range(width)
        width_range_O = width_range_E + 1

        # this transpose allows to only index the varying dimensions
        # only the first dimensions can be indexed in tensor_scatter_nd_add
        # we also need to match the indices with the updates reshaping
        scatter_nd_perm = [2, 1, 3, 0]
        outputs_reshaped = tf.transpose(outputs, perm=scatter_nd_perm)

        combos_list = [
            ((height_range_E, width_range_E), x_EE),
            ((height_range_O, width_range_E), x_OE),
            ((height_range_E, width_range_O), x_EO),
            ((height_range_O, width_range_O), x_OO),
        ]
        for (height_range, width_range), x_comb in combos_list:
            h_range, w_range = tf.meshgrid(height_range, width_range)
            h_range = tf.reshape(h_range, (-1,))
            w_range = tf.reshape(w_range, (-1,))
            combo_indices = tf.stack([w_range, h_range], axis=-1)
            combo_reshaped = tf.transpose(x_comb, perm=scatter_nd_perm)
            outputs_reshaped = tf.tensor_scatter_nd_add(
                outputs_reshaped,
                indices=combo_indices,
                updates=tf.reshape(combo_reshaped, (-1, batch_size, 1)),
            )

        inverse_scatter_nd_perm = [3, 1, 0, 2]
        outputs = tf.transpose(outputs_reshaped, perm=inverse_scatter_nd_perm)

        return outputs

class MWCNN(Model):
    def __init__(
            self,
            n_scales=3,
            kernel_size=3,
            bn=False,
            n_filters_per_scale=DEFAULT_N_FILTERS_PER_SCALE,
            n_convs_per_scale=DEFAULT_N_CONVS_PER_SCALE,
            n_first_convs=3,
            first_conv_n_filters=64,
            **kwargs,
        ):
        super(MWCNN, self).__init__(**kwargs)
        self.n_scales = n_scales
        self.kernel_size = kernel_size
        self.bn = bn
        self.n_filters_per_scale = n_filters_per_scale
        self.n_convs_per_scale = n_convs_per_scale
        self.n_first_convs = n_first_convs
        self.first_conv_n_filters = first_conv_n_filters
        if self.n_first_convs > 0:
            self.first_convs = [MWCNNConvBlock(
                n_filters=self.first_conv_n_filters,
                kernel_size=self.kernel_size,
                bn=self.bn,
            ) for _ in range(2 * self.n_first_convs)]
        self.conv_blocks_per_scale = [
            [MWCNNConvBlock(
                n_filters=self.n_filters_for_conv_for_scale(i_scale, i_conv),
                kernel_size=self.kernel_size,
                bn=self.bn,
            ) for i_conv in range(self.n_convs_per_scale[i_scale] * 2)]
            for i_scale in range(self.n_scales)
        ]
        # the last convolution is without bn and relu, and also has only
        # 4 filters, that's why we treat it separately
        self.conv_blocks_per_scale[0][-1] = Conv2D(
            4,
            self.kernel_size,
            padding='same',
            use_bias=True,
        )
        self.pooling = DWT()
        self.unpooling = IWT()

    def n_filters_for_conv_for_scale(self, i_scale, i_conv):
        n_filters = self.n_filters_per_scale[i_scale]
        if i_conv == self.n_convs_per_scale[i_scale] * 2 - 1:
            if i_scale == 0:
                n_filters = 4 * self.first_conv_n_filters
            else:
                n_filters = 4 * self.n_filters_per_scale[i_scale-1]
        return n_filters

    def call(self, inputs):
        last_feature_for_scale = []
        current_feature = inputs
        if self.n_first_convs > 0:
            for conv in self.first_convs[:self.n_first_convs]:
                current_feature = conv(current_feature)
            first_conv_feature = current_feature
        for i_scale in range(self.n_scales):
            current_feature = self.pooling(current_feature)
            n_convs = self.n_convs_per_scale[i_scale]
            for conv in self.conv_blocks_per_scale[i_scale][:n_convs]:
                current_feature = conv(current_feature)
            last_feature_for_scale.append(current_feature)
        for i_scale in range(self.n_scales - 1, -1, -1):
            if i_scale != self.n_scales - 1:
                current_feature = self.unpooling(current_feature)
                current_feature = current_feature + last_feature_for_scale[i_scale]
            n_convs = self.n_convs_per_scale[i_scale]
            for conv in self.conv_blocks_per_scale[i_scale][n_convs:]:
                current_feature = conv(current_feature)
        current_feature = self.unpooling(current_feature)
        if self.n_first_convs > 0:
            current_feature = current_feature + first_conv_feature
            for conv in self.first_convs[self.n_first_convs:]:
                current_feature = conv(current_feature)
        outputs = inputs + current_feature
        return outputs
