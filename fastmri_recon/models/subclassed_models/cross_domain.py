import tensorflow as tf
from tensorflow.keras.models import Model

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

    def call(self, inputs):
        # TODO: deal with the potential sensitivity maps
        kspace, mask = inputs
        kspace_buffer = tf.concat([kspace] * self.k_buffer_size, axis=-1)
        image = self.backward_operator(*inputs)
        image_buffer = tf.concat([image] * self.i_buffer_size, axis=-1)
        # TODO: create a buffer
        for i_domain, domain in enumerate(self.domain_sequence):
            if domain == 'K':
                if self.k_buffer_mode:
                    kspace_buffer = tf.concat([
                        kspace_buffer,
                        self.forward_operator(image_buffer, mask),
                    ], axis=-1)
                else:
                    kspace_buffer = self.forward_operator(image_buffer, mask)
                kspace_buffer = self.apply_data_consistency(kspace_buffer, *inputs)
                # NOTE: this i //2 suggest alternating domains, this will need
                # evolve if we want non-alternating domains. This needs to be
                # clear in the docs.
                kspace_buffer = self.kspace_net[i_domain//2](kspace_buffer)
            if domain == 'I':
                if self.i_buffer_mode:
                    image_buffer = tf.concat([
                        image_buffer,
                        self.backward_operator(kspace_buffer, mask),
                    ], axis=-1)
                else:
                    # NOTE: the operator is already doing the channel selection
                    image_buffer = self.backward_operator(kspace_buffer, mask)
                image_buffer = self.image_net[i_domain//2](image_buffer)
        image = tf_fastmri_format(image_buffer[..., 0:1])
        return image

    def apply_data_consistency(self, kspace, original_kspace, mask):
        if self.data_consistency_mode == 'measurements_residual':
            return tf.concat([kspace, original_kspace], axis=-1)
        else:
            return _replace_values_on_mask([kspace, original_kspace, mask])

    def forward_operator(self, image, mask):
        if self.data_consistency_mode == 'measurements_residual':
            # TODO: when dealing with non cartesian/pMRI change this to self.op
            # defined in init
            return self.op([image, mask])
        else:
            return self.op(image)

    def backward_operator(self, kspace, mask):
        if self.data_consistency_mode == 'measurements_residual':
            return self.adj_op([kspace, mask])
        else:
            return self.adj_op(kspace)
