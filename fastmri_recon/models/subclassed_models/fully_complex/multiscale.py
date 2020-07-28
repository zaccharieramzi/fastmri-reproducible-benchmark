import tensorflow as tf
from tensorflow.keras.models import Model

from ...utils.pad_for_pool import pad_for_pool


class Multiscale(Model):
    """A wrapper model around a multiscale complex-valued network.

    This class allows to "decorate" a multiscale complex-valued network, in a
    model accepting output of different input sizes.
    It pads the input according to the number of scales present in the
    multiscale network to allow for handling images that do not have a power of
    2 shape.

    Parameters:
        model (tf.keras.models.Model): the multiscale float-valued network.
        res (bool): whether you want to add a residual connection to the network
            that only takes into account `n_output_channels` channel elements
            of the input. Defaults to False.
        n_scales (int): the number of scales in the multiscale float-valued
            network. Defaults to 0.
        n_output_channels (int): the number of expected output channels. Defaults
            to 1.
        **kwargs: keyword arguments to tf.keras.models.Model.
    """
    def __init__(
            self,
            model,
            res=False,
            n_scales=0,
            n_output_channels=1,
            **kwargs,
        ):
        super(Multiscale, self).__init__(**kwargs)
        self.model = model
        self.res = res
        self.n_scales = n_scales
        self.n_output_channels = n_output_channels

    def call(self, inputs):
        outputs = inputs
        if self.n_scales > 0:
            outputs, n_pad = pad_for_pool(inputs, self.n_scales)
        outputs = self.model(outputs)
        if self.n_scales > 0:
            outputs = tf.cond(
                n_pad == 0,
                lambda: outputs,
                lambda: outputs[:, :, n_pad//2:-n_pad//2],
            )
        if self.res:
            outputs = inputs[..., :self.n_output_channels] + outputs
        return outputs
