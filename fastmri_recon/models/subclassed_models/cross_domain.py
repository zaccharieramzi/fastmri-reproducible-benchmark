import tensorflow as tf
from tensorflow.keras.models import Model

from .unet import  UnetComplex
from ..utils.data_consistency import _replace_values_on_mask
from ..utils.fastmri_format import tf_fastmri_format


class CrossDomainNet(Model):
    def __init__(
            self,
            domain_sequence='KIKI',
            data_consistency_mode='measurements_residual',
            i_buffer_mode=False,
            k_buffer_mode=False,
            i_buffer_size=1,
            k_buffer_size=1,
            multicoil=False,
            refine_smaps=False,
            normalize_image=False,
            fastmri=True,
            **kwargs,
        ):
        super(CrossDomainNet, self).__init__(**kwargs)
        self.domain_sequence = domain_sequence
        self.data_consistency_mode = data_consistency_mode
        self.i_buffer_mode = i_buffer_mode
        self.k_buffer_mode = k_buffer_mode
        # TODO: if not buffer mode set to 1 both
        self.i_buffer_size = i_buffer_size
        self.k_buffer_size = k_buffer_size
        self.multicoil = multicoil
        self.refine_smaps = refine_smaps
        self.normalize_image = normalize_image
        self.fastmri = fastmri
        if self.multicoil and self.refine_smaps:
            self.smaps_refiner = UnetComplex(
                n_layers=3,
                layers_n_channels=[4 * 2**i for i in range(3)],
                layers_n_non_lins=2,
                n_input_channels=1,
                n_output_channels=1,
                res=True,
                non_linearity='lrelu',
                channel_attention_kwargs=None,
                name=f'smaps_refiner',
            )

    def call(self, inputs):
        if self.multicoil:
            original_kspace, mask, smaps = inputs
            if self.refine_smaps:
                # we deal with each smap independently
                smaps_shape = tf.shape(smaps)
                batch_size = smaps_shape[0]
                n_coils = smaps_shape[1]
                smaps_contig = tf.reshape(
                    smaps,
                    [batch_size * n_coils, smaps_shape[2], smaps_shape[3], 1],
                )
                smaps_refined = self.smaps_refiner(smaps_contig)
                smaps_refined = tf.reshape(
                    smaps_refined,
                    [batch_size, n_coils, smaps_shape[2], smaps_shape[3]],
                )
                rss = tf.norm(smaps_refined, axis=1, keepdims=True)
                smaps_refined_normalized = smaps_refined / rss
                smaps = smaps_refined_normalized
            # TODO: change when doing non uniform multicoil
            op_args = ()
        else:
            if len(inputs) == 2:
                original_kspace, mask = inputs
                op_args = ()
            else:
                original_kspace, mask, op_args = inputs
            smaps = None
        image = self.backward_operator(original_kspace, mask, smaps, *op_args)
        if self.normalize_image:
            normalization_factor = tf.reduce_max(
                tf.abs(image),
                axis=[1, 2, 3, 4] if self.multicoil else [1, 2, 3],
            )
            normalization_factor = tf.cast(normalization_factor, image.dtype)
            orig_image_shape = image.shape
            orig_kspace_shape = original_kspace.shape
            image = image / normalization_factor
            image.set_shape(orig_image_shape)
            original_kspace = original_kspace / normalization_factor
            original_kspace.set_shape(orig_kspace_shape)
        kspace_buffer = tf.concat([original_kspace] * self.k_buffer_size, axis=-1)
        image_buffer = tf.concat([image] * self.i_buffer_size, axis=-1)
        for i_domain, domain in enumerate(self.domain_sequence):
            if domain == 'K':
                forward_op_res = self.forward_operator(image_buffer, mask, smaps)
                if isinstance(forward_op_res, tuple):
                    forward_op_res = forward_op_res[0]
                if self.k_buffer_mode:
                    kspace_buffer = tf.concat([
                        kspace_buffer,
                        forward_op_res,
                    ], axis=-1)
                else:
                    kspace_buffer = forward_op_res
                kspace_buffer = self.apply_data_consistency(kspace_buffer, original_kspace, mask)
                # NOTE: this i //2 suggest alternating domains, this will need
                # evolve if we want non-alternating domains. This needs to be
                # clear in the docs.
                kspace_buffer = self.kspace_net[i_domain//2](kspace_buffer)
            if domain == 'I':
                if self.i_buffer_mode:
                    backward_op_res = self.backward_operator(kspace_buffer, mask, smaps, *op_args)
                    if self.normalize_image:
                        normalization_factor_iteration = tf.reduce_max(
                            tf.abs(backward_op_res),
                            axis=[1, 2, 3, 4] if self.multicoil else [1, 2, 3],
                        )
                        orig_bopres_shape = backward_op_res.shape
                        normalization_factor_iteration = tf.cast(normalization_factor_iteration, image.dtype)
                        backward_op_res = backward_op_res / normalization_factor_iteration
                        backward_op_res.set_shape(orig_bopres_shape)
                    image_buffer = tf.concat([
                        image_buffer,
                        backward_op_res,
                    ], axis=-1)
                else:
                    # NOTE: the operator is already doing the channel selection
                    image_buffer = self.backward_operator(kspace_buffer, mask, smaps, *op_args)
                image_buffer = self.image_net[i_domain//2](image_buffer)
        # if self.normalize_image:
        #     image_buffer = image_buffer * normalization_factor
        if self.fastmri:
            image = tf_fastmri_format(image_buffer[..., 0:1])
        else:
            image = tf.abs(image_buffer[..., 0:1])
        return image

    def apply_data_consistency(self, kspace, original_kspace, mask):
        if self.data_consistency_mode == 'measurements_residual':
            return tf.concat([kspace, original_kspace], axis=-1)
        else:
            return _replace_values_on_mask([kspace, original_kspace, mask])

    def forward_operator(self, image, mask, smaps):
        if self.data_consistency_mode == 'measurements_residual':
            if self.multicoil:
                return self.op([image, mask, smaps])
            else:
                return self.op([image, mask])
        else:
            if self.multicoil:
                return self.op([image, smaps])
            else:
                return self.op(image)

    def backward_operator(self, kspace, mask, smaps, *op_args):
        if self.data_consistency_mode == 'measurements_residual':
            if self.multicoil:
                return self.adj_op([kspace, mask, smaps, *op_args])
            else:
                return self.adj_op([kspace, mask, *op_args])
        else:
            if self.multicoil:
                return self.adj_op([kspace, smaps, *op_args])
            else:
                return self.adj_op(kspace)
