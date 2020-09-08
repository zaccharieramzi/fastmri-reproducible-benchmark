"""From https://github.com/zaccharieramzi/tf-focnet/blob/master/focnet.py
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    BatchNormalization,
    Activation,
    AveragePooling2D,
    Conv2DTranspose,
)


DEFAULT_N_CONVS_PER_SCALE = [5, 11, 11, 7]
DEFAULT_COMMUNICATION_BETWEEN_SCALES = [
    [(1, 0), (1, 2), (2, 3), (2, 5), (3, 6), (3, 8), (4, 9), (4, 11)],  # 1-2
    [(1, 0), (1, 2), (4, 3), (4, 5), (7, 6), (7, 8), (10, 9), (10, 11)],  # 2-3
    [(1, 0), (1, 1), (4, 2), (4, 3), (7, 4), (7, 5), (10, 6), (10, 7)],  # 3-4
]


def residual_weights_computation(t, beta):
    w = [beta]
    for k in range(t-1, 0, -1):
        w_k = (1 - (1 + beta) / (t - k + 1)) * w[-1]
        w.append(w_k)
    w = w[::-1]
    return w


class FocConvBlock(Layer):
    def __init__(self, n_filters=128, kernel_size=3, bn=True, **kwargs):
        super(FocConvBlock, self).__init__(**kwargs)
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

    def get_config(self):
        config = super(FocConvBlock, self).get_config()
        bn = not(self.bn is None)
        config.update({
            'bn': bn,
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
        })
        return config

class SwitchLayer(Layer):
    # form what I understand from the code this is what a switch layer looks
    # like
    # very difficult to read from this
    # https://github.com/hsijiaxidian/FOCNet/blob/master/%2Bdagnn/Gate.m
    # which is used here
    # https://github.com/hsijiaxidian/FOCNet/blob/master/FracDCNN.m#L360
    def __init__(self, **kwargs):
        super(SwitchLayer, self).__init__(**kwargs)
        self.switch = self.add_weight(
            'switch_' + str(K.get_uid('switch')),
            shape=(),
            initializer=tf.constant_initializer(2),  # we add a big initializer
            # to take into account the adjacent scales by default
            # but not too big because we want to have some gradient flowing
        )

    def call(self, inputs):
        outputs = inputs * tf.sigmoid(self.switch)
        return outputs

class FocNet(Model):
    def __init__(
            self,
            n_scales=4,
            n_filters=128,
            kernel_size=3,
            bn=False,
            n_convs_per_scale=DEFAULT_N_CONVS_PER_SCALE,
            communications_between_scales=DEFAULT_COMMUNICATION_BETWEEN_SCALES,
            beta=0.2,
            n_outputs=1,
            **kwargs,
        ):
        super(FocNet, self).__init__(**kwargs)
        self.n_scales = n_scales
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.bn = bn
        self.n_convs_per_scale = n_convs_per_scale
        self.communications_between_scales = communications_between_scales
        self.beta = beta
        self.n_outputs = n_outputs
        self.pooling = AveragePooling2D(padding='same')
        self.unpoolings_per_scale = [
            [
                Conv2DTranspose(
                    self.n_filters,
                    self.kernel_size,
                    strides=(2, 2),
                    padding='same',
                )
                for _ in range(len(self.communications_between_scales[i_scale])//2)
            ]
            for i_scale in range(self.n_scales - 1)
        ]
        # unpooling is not specified in the paper, but in the code
        # you can see a deconv is used
        # https://github.com/hsijiaxidian/FOCNet/blob/master/FracDCNN.m#L415
        self.n_switches_per_scale = []
        self.compute_n_switches_per_scale()
        self.switches_per_scale = [
            [
                SwitchLayer()
                for _ in range(self.n_switches_per_scale[i_scale])
            ]
            for i_scale in range(self.n_scales)
        ]
        self.first_conv = Conv2D(
            self.n_filters,  # we output a grayscale image
            self.kernel_size,  # we simply do a linear combination of the features
            padding='same',
            use_bias=True,
        )
        self.conv_blocks_per_scale = [
            [FocConvBlock(
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                bn=self.bn,
            ) for _ in range(n_conv_blocks)]
            for n_conv_blocks in self.n_convs_per_scale
        ]
        self.final_conv = Conv2D(
            self.n_outputs,
            1,  # we simply do a linear combination of the features
            padding='same',
            use_bias=True,
        )
        self.needs_to_compute = {}
        self.build_needs_to_compute()

    def build_needs_to_compute(self):
        for i_scale, scale_communication in enumerate(self.communications_between_scales):
            down = True
            for i_conv_scale_up, i_conv_scale_down in scale_communication:
                scale_up_node = (i_scale, i_conv_scale_up)
                scale_down_node = (i_scale + 1, i_conv_scale_down)
                if down:
                    self.needs_to_compute[scale_down_node] = scale_up_node
                else:
                    self.needs_to_compute[scale_up_node] = scale_down_node
                down = not down

    def compute_n_switches_per_scale(self):
        for i_scale in range(self.n_scales):
            if i_scale == 0:
                n_switches = len(self.communications_between_scales[0]) // 2
            elif i_scale == self.n_scales - 1:
                n_switches = len(self.communications_between_scales[-1]) // 2
            else:
                n_switches = len(self.communications_between_scales[i_scale - 1]) // 2
                n_switches += len(self.communications_between_scales[i_scale]) // 2
            self.n_switches_per_scale.append(n_switches)


    def call(self, inputs):
        features_per_scale = [[] for _ in range(self.n_scales)]
        features_per_scale[0].append(self.first_conv(inputs))
        unpoolings_used_per_scale = [0 for _ in range(self.n_scales - 1)]
        switches_used_per_scale = [0 for _ in range(self.n_scales)]
        i_scale = 0
        i_feature = 0
        while i_scale != 0 or i_feature < self.n_convs_per_scale[0]:
            if i_feature >= self.n_convs_per_scale[i_scale]:
                i_scale -= 1
                i_feature = len(features_per_scale[i_scale]) - 1
            node_to_compute = self.needs_to_compute.get(
                (i_scale, i_feature),
                None,
            )
            if node_to_compute is not None:
                i_scale_to_compute, i_feature_to_compute = node_to_compute
                # test if feature is already computed
                n_features_scale_to_compute = len(features_per_scale[i_scale_to_compute])
                if n_features_scale_to_compute <= i_feature_to_compute:
                    # the feature has not been computed, we need to compute it
                    i_scale = i_scale_to_compute
                    i_feature = max(n_features_scale_to_compute - 1, 0)
                    # if there are no features we add it as well
                    continue
                else:
                    # the feature has already been computed we can just use it as is
                    additional_feature = features_per_scale[i_scale_to_compute][i_feature_to_compute]
                    if i_scale_to_compute > i_scale:
                        # the feature has to be unpooled and switched
                        # for now since I don't understand switching, I just do
                        # unpooling, switching will be implemented later on
                        i_unpooling = unpoolings_used_per_scale[i_scale]
                        unpooling = self.unpoolings_per_scale[i_scale][i_unpooling]
                        additional_feature_processed = unpooling(
                            additional_feature,
                        )
                        unpoolings_used_per_scale[i_scale] += 1
                    else:
                        # the feature has to be pooled
                        additional_feature_processed = self.pooling(
                            additional_feature,
                        )
                    i_switch = switches_used_per_scale[i_scale]
                    switch = self.switches_per_scale[i_scale][i_switch]
                    additional_feature_processed = switch(additional_feature_processed)
                    switches_used_per_scale[i_scale] += 1
                    if len(features_per_scale[i_scale]) == 0:
                        # this is the first feature added to the scale
                        features_per_scale[i_scale].append(additional_feature_processed)
                        feature = additional_feature_processed
                    else:
                        feature = tf.concat([
                            features_per_scale[i_scale][i_feature],
                            additional_feature_processed,
                        ], axis=-1)

            else:
                feature = features_per_scale[i_scale][-1]
            new_feature = self.conv_blocks_per_scale[i_scale][i_feature](
                feature
            )
            weights = residual_weights_computation(
                i_feature,
                beta=self.beta,
            )
            for weight, res_feature in zip(weights, features_per_scale[i_scale]):
                new_feature = new_feature + weight * res_feature
            features_per_scale[i_scale].append(new_feature)
            i_feature += 1
        outputs = self.final_conv(features_per_scale[0][self.n_convs_per_scale[0]])
        # this could be -1 instead of self.n_convs_per_scale[0], but it's an
        # extra sanity check that everything is going alright
        return outputs

    def get_config(self):
        config = super(FocNet, self).get_config()
        config.update({
            'n_scales': self.n_scales,
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'bn': self.bn,
            'n_convs_per_scale': self.n_convs_per_scale,
            'communications_between_scales': self.communications_between_scales,
            'beta': self.beta,
            'n_outputs': self.n_outputs,
        })
        return config
