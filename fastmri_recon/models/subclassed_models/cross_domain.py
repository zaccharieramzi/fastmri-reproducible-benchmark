import tensorflow as tf
from tensorflow.keras.models import Model

from ..utils.data_consistency import _replace_values_on_mask
from ..utils.fastmri_format import tf_fastmri_format
from ..utils.fourier import tf_op, tf_adj_op, tf_unmasked_op, tf_unmasked_adj_op


class CrossDomainNet(Model):
    def __init__(self, data_consistency_mode='measurements_residual', **kwargs):
        super(CrossDomainNet, self).__init__(**kwargs)
        self.data_consistency_mode = data_consistency_mode

    def call(self, inputs):
        # TODO: deal with the potential sensitivity maps
        kspace, mask = inputs
        # TODO: create a buffer
        for i_iter in range(self.n_iter):
            kspace = self.kspace_net(kspace)
            image = self.backward_operator(kspace, mask)
            image = self.image_net(image)
            kspace = self.forward_operator(image, mask)
            kspace = self.apply_data_consistency(kspace, *inputs)
        image = self.pseudo_inverse(kspace)
        image = tf_fastmri_format(image)
        return image

    def apply_data_consistency(self, kspace, original_kspace, mask):
        if self.data_consistency_mode == 'measurements_residual':
            return kspace
        else:
            return _replace_values_on_mask([kspace, original_kspace, mask])

    def forward_operator(self, image, mask):
        if self.data_consistency_mode == 'measurements_residual':
            # TODO: when dealing with non cartesian/pMRI change this to self.op
            # defined in init
            return tf_op([image, mask])
        else:
            return tf_unmasked_op(image)

    def backward_operator(self, kspace, mask):
        if self.data_consistency_mode == 'measurements_residual':
            return tf_adj_op([kspace, mask])
        else:
            return tf_unmasked_adj_op(kspace)
