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
        if self.refine_smaps:
            self.smaps_refiner = UnetComplex(
                n_layers=3,
                layers_n_channels=[4 * 2**i for i in range(3)],
                layers_n_non_lins=2,
                n_input_channels=2,
                n_output_channels=2,
                res=True,
                non_linearity='lrelu',
                channel_attention_kwargs=None,
                name=f'smaps_refiner',
            )

    def call(self, inputs):
        # TODO: deal with the potential sensitivity maps
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
        else:
            original_kspace, mask = inputs
            smaps = None
        kspace_buffer = tf.concat([original_kspace] * self.k_buffer_size, axis=-1)
        image = self.backward_operator(original_kspace, mask, smaps)
        image_buffer = tf.concat([image] * self.i_buffer_size, axis=-1)
        # TODO: create a buffer
        for i_domain, domain in enumerate(self.domain_sequence):
            if domain == 'K':
                if self.k_buffer_mode:
                    kspace_buffer = tf.concat([
                        kspace_buffer,
                        self.forward_operator(image_buffer, mask, smaps),
                    ], axis=-1)
                else:
                    kspace_buffer = self.forward_operator(image_buffer, mask, smaps)
                kspace_buffer = self.apply_data_consistency(kspace_buffer, original_kspace, mask)
                # NOTE: this i //2 suggest alternating domains, this will need
                # evolve if we want non-alternating domains. This needs to be
                # clear in the docs.
                kspace_buffer = self.kspace_net[i_domain//2](kspace_buffer)
            if domain == 'I':
                if self.i_buffer_mode:
                    image_buffer = tf.concat([
                        image_buffer,
                        self.backward_operator(kspace_buffer, mask, smaps),
                    ], axis=-1)
                else:
                    # NOTE: the operator is already doing the channel selection
                    image_buffer = self.backward_operator(kspace_buffer, mask, smaps)
                image_buffer = self.image_net[i_domain//2](image_buffer)
        image = tf_fastmri_format(image_buffer[..., 0:1])
        return image

    def apply_data_consistency(self, kspace, original_kspace, mask):
        if self.data_consistency_mode == 'measurements_residual':
            return tf.concat([kspace, original_kspace], axis=-1)
        else:
            return _replace_values_on_mask([kspace, original_kspace, mask])

    def forward_operator(self, image, mask, smaps):
        if self.data_consistency_mode == 'measurements_residual':
            # TODO: when dealing with non cartesian/pMRI change this to self.op
            # defined in init
            if self.multicoil:
                return self.op([image, mask, smaps])
            else:
                return self.op([image, mask])
        else:
            if self.multicoil:
                return self.op([image, smaps])
            else:
                return self.op(image)

    def backward_operator(self, kspace, mask, smaps):
        if self.data_consistency_mode == 'measurements_residual':
            if self.multicoil:
                return self.adj_op([kspace, mask, smaps])
            else:
                return self.adj_op([kspace, mask])
        else:
            if self.multicoil:
                return self.adj_op([kspace, smaps])
            else:
                return self.adj_op(kspace)
