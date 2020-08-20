"""From https://github.com/zaccharieramzi/tf-didn
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, PReLU, Layer
from tensorflow.keras.models import Model


CONVS_PER_SCALE_DUB = [
    (2, 2),
    (1, 1),
    (1,),
]
N_SCALES_DUB = 3

class Subpixel(Conv2D):
    # loosely inspired by https://github.com/atriumlts/subpixel/blob/master/keras_subpixel.py
    def __init__(self, filters, r, **kwargs):
        kwargs.pop('dilation_rate', None)
        super(Subpixel, self).__init__(filters=(r**2)*filters, **kwargs)
        self.r = r

    def call(self, inputs):
        conv_output = super(Subpixel, self).call(inputs)
        upscaled_output = tf.nn.depth_to_space(conv_output, self.r)
        return upscaled_output

    def get_config(self):
        config = super(Subpixel, self).get_config()
        r = self.r
        filters = self.filters // r**2
        config.update({
            'r': r,
            'filters': filters,
        })
        return config

class DUB(Model):
    def __init__(
            self,
            n_filters=256,
            n_scales=N_SCALES_DUB,
            convs_per_scale=CONVS_PER_SCALE_DUB,
            **kwargs,
        ):
        super(DUB, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.n_scales = n_scales
        self.convs_per_scale = convs_per_scale
        self.poolings = [
            Conv2D(
                filters=self.n_filters*2**(i_scale+1),
                kernel_size=3,
                strides=2,
                padding='same',
                use_bias=True,
            ) for i_scale in range(self.n_scales - 1)
        ]
        self.unpoolings = [
            Subpixel(
                filters=self.n_filters*2**i_scale,
                kernel_size=1,
                r=2,
                padding='same',
                use_bias=True,
            ) for i_scale in range(self.n_scales - 1)
        ]
        self.convs = [
            [
                Conv2D(
                    filters=self.n_filters*2**i_scale,
                    kernel_size=3,
                    padding='same',
                    activation=PReLU(shared_axes=[1, 2]),
                    use_bias=True,
                ) for _ in range(sum(self.convs_per_scale[i_scale]))
            ]
            for i_scale in range(self.n_scales)
        ]
        self.agg_feature_maps = [
            Conv2D(
                filters=self.n_filters*2**i_scale,
                kernel_size=1,
                padding='same',
                use_bias=True,
            ) for i_scale in range(self.n_scales - 1)
        ]
        self.final_conv = Conv2D(
            filters=self.n_filters,
            kernel_size=3,
            padding='same',
            activation=PReLU(shared_axes=[1, 2]),
            use_bias=True,
        )


    def call(self, inputs):
        outputs = inputs
        scale_outputs = []
        for i_scale in range(self.n_scales):
            scale_input = outputs
            n_convs = self.convs_per_scale[i_scale][0]
            for conv in self.convs[i_scale][:n_convs]:
                outputs = conv(outputs)
            outputs = outputs + scale_input
            if i_scale < self.n_scales - 1:
                scale_outputs.append(outputs)
                outputs = self.poolings[i_scale](outputs)
        for i_scale in range(self.n_scales - 2, -1, -1):
            outputs = self.unpoolings[i_scale](outputs)
            outputs = tf.concat([outputs, scale_outputs[i_scale]], axis=-1)
            outputs = self.agg_feature_maps[i_scale](outputs)
            scale_input = outputs
            n_convs = self.convs_per_scale[i_scale][0]
            for conv in self.convs[i_scale][n_convs:]:
                outputs = conv(outputs)
            outputs = outputs + scale_input
        outputs = self.final_conv(outputs)
        outputs = outputs + inputs
        return outputs

    def get_config(self):
        config = super(DUB, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'n_scales': self.n_scales,
            'convs_per_scale': self.convs_per_scale,
        })
        return config

class ReconBlock(Layer):
    def __init__(self, n_convs=9, n_filters=256, **kwargs):
        super(ReconBlock, self).__init__(**kwargs)
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.convs = [
            Conv2D(
                filters=self.n_filters,
                kernel_size=3,
                padding='same',
                activation=PReLU(shared_axes=[1, 2]),
                use_bias=True,
            ) for _ in range(self.n_convs - 1)
        ]
        self.convs.append(Conv2D(
            filters=self.n_filters,
            kernel_size=3,
            padding='same',
            use_bias=True,
        ))

    def call(self, inputs):
        outputs = inputs
        for conv in self.convs:
            outputs = conv(outputs)
        outputs = outputs + inputs
        return outputs

    def get_config(self):
        config = super(ReconBlock, self).get_config()
        config.update({
            'n_convs': self.n_convs,
            'n_filters': self.n_filters,
        })
        return config


class DIDN(Model):
    def __init__(
            self,
            n_filters=256,
            # the number of dubs is inferred from the code not the paper
            n_dubs=6,
            n_convs_recon=9,
            n_scales=3,
            convs_per_scale=CONVS_PER_SCALE_DUB,
            n_outputs=1,
            res=False,
            **kwargs
        ):
        super(DIDN, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.n_dubs = n_dubs
        self.n_scales = n_scales
        self.convs_per_scale = convs_per_scale
        self.n_convs_recon = n_convs_recon
        self.n_outputs = n_outputs
        self.res = res
        self.dubs = [
            DUB(
                n_filters=self.n_filters,
                n_scales=self.n_scales,
                convs_per_scale=self.convs_per_scale,
            ) for _ in range(self.n_dubs)
        ]
        self.recon_block = ReconBlock(
            n_convs=self.n_convs_recon,
            n_filters=self.n_filters,
        )
        self.first_conv = Conv2D(
            filters=self.n_filters,
            kernel_size=3,
            padding='same',
            activation=PReLU(shared_axes=[1, 2]),
            use_bias=True,
        )
        self.pooling = Conv2D(
            filters=self.n_filters,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=True,
        )
        self.post_recon_agg = Conv2D(
            filters=self.n_filters,
            kernel_size=1,
            padding='same',
            use_bias=True,
        )
        self.post_recon_conv = Conv2D(
            filters=self.n_filters,
            kernel_size=3,
            padding='same',
            activation=PReLU(shared_axes=[1, 2]),
            use_bias=True,
        )
        self.last_conv = Conv2D(
            filters=self.n_outputs,
            kernel_size=3,
            padding='same',
            # in code the activation is actually linear
            activation='linear',
            use_bias=True,
        )

    def call(self, inputs):
        outputs = self.first_conv(inputs)
        outputs = self.pooling(outputs)
        dub_outputs = []
        for dub in self.dubs:
            outputs = dub(outputs)
            dub_outputs.append(outputs)
        recon_outputs = [
            self.recon_block(dub_output) for dub_output in dub_outputs
        ]
        recon_outputs = tf.concat(recon_outputs, axis=-1)
        recon_agg = self.post_recon_agg(recon_outputs)
        outputs = self.post_recon_conv(recon_agg)
        outputs = tf.nn.depth_to_space(outputs, 2)
        outputs = self.last_conv(outputs)
        if self.res:
            outputs = outputs + inputs
        return outputs

    def get_config(self):
        config = super(DIDN, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'n_dubs': self.n_dubs,
            'n_scales': self.n_scales,
            'convs_per_scale': self.convs_per_scale,
            'n_convs_recon': self.n_convs_recon,
            'n_outputs': self.n_outputs,
            'res': self.res,
        })
        return config
