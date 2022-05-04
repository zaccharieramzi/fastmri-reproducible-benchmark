import tensorflow as tf
from tensorflow.keras.models import Model

from ..functional_models.unet import unet
from ..utils.complex import to_complex
from ..utils.fastmri_format import general_fastmri_format
from ..utils.fourier import AdjNFFT
from ..utils.pad_for_pool import pad_for_pool


class UnetComplex(Model):
    def __init__(
            self,
            n_input_channels=1,
            n_output_channels=1,
            kernel_size=3,
            n_layers=1,
            layers_n_channels=1,
            layers_n_non_lins=1,
            res=False,
            non_linearity='relu',
            channel_attention_kwargs=None,
            dealiasing_nc_fastmri=False,
            im_size=None,
            dcomp=None,
            multicoil=False,
            grad_traj=False,
            nufft_implementation='tfkbnufft',
            **kwargs,
        ):
        super(UnetComplex, self).__init__(**kwargs)
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.layers_n_channels = layers_n_channels
        self.layers_n_non_lins = layers_n_non_lins
        self.res = res
        self.non_linearity = non_linearity
        if channel_attention_kwargs is None:
            self.channel_attention_kwargs = {}
        else:
            self.channel_attention_kwargs = channel_attention_kwargs
        self.dealiasing_nc_fastmri = dealiasing_nc_fastmri
        if self.dealiasing_nc_fastmri:
            self.multicoil = multicoil
            self.adj_op = AdjNFFT(
                im_size=im_size,
                multicoil=self.multicoil,
                density_compensation=dcomp,
                grad_traj=grad_traj,
                implementation=nufft_implementation,
            )
        self.unet = unet(
            input_size=(None, None, 2 * self.n_input_channels),  # 2 for real and imag
            n_output_channels=2 * self.n_output_channels,
            kernel_size=self.kernel_size,
            n_layers=self.n_layers,
            layers_n_channels=self.layers_n_channels,
            layers_n_non_lins=self.layers_n_non_lins,
            non_relu_contract=False,
            non_linearity=self.non_linearity,
            pool='max',
            compile=False,
            **self.channel_attention_kwargs,
        )

    def call(self, inputs):
        output_shape = None
        if self.dealiasing_nc_fastmri:
            if self.multicoil:
                if len(inputs) == 3:
                    original_kspace, mask, smaps = inputs
                    op_args = ()
                elif len(inputs) == 4:
                    original_kspace, mask, smaps, op_args = inputs
                else:
                    original_kspace, mask, smaps, output_shape, op_args = inputs
                outputs = self.adj_op([original_kspace, mask, smaps, *op_args])
            else:
                if len(inputs) == 2:
                    original_kspace, mask = inputs
                    op_args = ()
                else:
                    original_kspace, mask, op_args = inputs
                outputs = self.adj_op([original_kspace, mask, *op_args])
            # we do this to match the residual part.
            inputs = outputs
        else:
            outputs = inputs
        outputs, padding = pad_for_pool(outputs, self.n_layers-1)
        outputs = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], axis=-1)
        outputs = self.unet(outputs)
        outputs = to_complex(outputs, self.n_output_channels)
        outputs = tf.cond(
            tf.reduce_sum(padding) == 0,
            lambda: outputs,
            lambda: outputs[:, :, padding[0]:-padding[1]],
        )
        if self.res:
            outputs = inputs[..., :self.n_output_channels] + outputs
        if self.dealiasing_nc_fastmri:
            outputs = general_fastmri_format(outputs, output_shape=output_shape)
        return outputs
