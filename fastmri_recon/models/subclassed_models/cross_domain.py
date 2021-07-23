from contextlib import ExitStack

import tensorflow as tf
from tensorflow.keras.models import Model

from .unet import  UnetComplex
from ..utils.data_consistency import _replace_values_on_mask
from ..utils.fastmri_format import general_fastmri_format
from ..utils.gpu_placement import gpu_index_from_submodel_index, get_gpus


class CrossDomainNet(Model):
    r"""Cross Domain network as defined in [R2020].

    This is an skeleton class implementing most of the logic for so-called
    cross-domain/unrolled networks. It basically alternates between the image
    and the measurements domains using a forward operator and its adjoint.
    It also performs data consistency in the form of residual or replacement.

    To implement a class inheriting from this class you need to have different
    attributes:
        kspace_net (tf.keras.models.Model): the k-space (measurements)
            correction network. This is where you would typically implement the
            residual in the measurements. In case of multicoil don't forget to
            account for the coil dimension.
            The input and output tensors must be tf.complex64.
            Input: nslices x (ncoils if multicoil) x spatial dimensions x (k_buffer_size + 2)
            Output: nslices x (ncoils if multicoil) x spatial dimensions x k_buffer_size
            If the data consistency mode is not 'measurements_residual', then
            the channel size of the input is actually (k_buffer_size + 2).
            If no buffer is used for kspace, the last dimensions for input and
            outputs are 2 (1 when not using 'measurements_residual') and 1.
        image_net (tf.keras.models.Model): the image space correction network.
            Don't forget to account for varying input shapes.
            The input and output tensors must be tf.complex64.
            Input: nslices x spatial dimensions x (i_buffer_size + 1)
            Output: nslices x spatial dimensions x i_buffer_size
            If no buffer is use for image space, the last dimensions are 1.
        op (tf.keras.layers.Layer): the forward operator, typically a Fourier
            transform. It takes in a list of tensors composed of (in this order):
                - image (tf.complex64): nslices x spatial dimensions x i_buffer_size
                - mask (type determined at runtime): dimensions can be
                    determined by input. This is present only when
                    the data consistency mode is residual.
                - smaps (tf.complex64): nslices x ncoils x spatial dimensions.
                    This is present only when `multicoil` is True.
        adj_op (tf.keras.layers.Layer): the adjoint operator, typically an
            adjoint Fourier transform. It takes in a list of tensors composed of
            (in this order):
                - kspace (tf.complex64): nslices x ncoils x spatial dimensions x k_buffer_size
                - mask (type determined at runtime): dimensions can be
                    determined by input. This is present only when
                    the data consistency mode is residual.
                - smaps (tf.complex64): nslices x ncoils x spatial dimensions.
                    This is present only when `multicoil` is True.
                - *op_args (tuple): optional extra arguments for the operator.
                    They are given as input to the model, and must be the same
                    for a given input (i.e. not change accross the model
                    iterations).

    Parameters:
        domain_sequence (str): the alternation sequence between kspace and image
            space. Currently, becaue of issue #82, it's not possible to use with
            anything other than real alternating sequence.
            For example you could have `domain_sequence='KIKIKI'`, to specify
            a cross domain network alternating between kspace and image space
            three times and starting with kspace. Defaults to 'KIKI'.
        data_consistency_mode (str): 'measurements_residual' or 'replacement'.
            When you use 'measurements_residual', the input to the kspace net
            will feature the original kspace.
            When using 'replacement', the kspace values at sampled positions
            are replaced by the original kspace values before being corrected
            by the kspace net. Defaults to 'measurements_residual'.
        i_buffer_mode (bool): whether you want to use a buffer for the image
            space. See [A2017] for more details on buffers. Defaults to False.
        k_buffer_mode (bool): whether you want to use a buffer for the kspace
            See [A2017] for more details on buffers. Defaults to False.
        i_buffer_size (int): the size of the buffer in the image space. Not
            taken into account when i_buffer_mode is False. Defaults to 1.
        k_buffer_size (int): the size of the buffer in the kspace. Not
            taken into account when k_buffer_mode is False. Defaults to 1.
        multicoil (bool): whether the input data is multicoil. Defaults to False.
        refine_smaps (bool): whether you want to refine the sensitivity maps
            with a neural network. The neural network applies the same function
            to each coil. For more details on this see [S2020].
            The neural network employed here is a U-net with 3 scales, leaky
            ReLU non-linearity, 4 base filters and a residual connection.
            Not taken into account when multicoil is False. Defaults to False.
        normalize_image (bool): whether you want to divide the image by its
            maximum value before it is fed in the image net. This is for example
            useful when you have high density in the middle of the kspace.
        multi_gpu (bool): whether you want to place the different iteration
            blocks on different GPUs. Only works with real alter sequences.
            Defaults to False.
        output_shape_spec (bool): whether the output shape is present in the
            input. This is taken into account only in multicoil and cartesian.
            Defaults to False.
        **kwargs: tf.keras.models.Model keyword arguments.

    Attributes:
        n_iter (int): used when `multi_gpu` is True. This allows to determine
            how many blocks are in the model.
        smaps_refiner (tf.keras.models.Model): the neural network responsible
            for refining the sensitivity maps. Exists only if `multicoil` and
            `refine_smaps` are True.
        available_gpus (list of str): the names of the available gpus. Exists
            only if `multi_gpu` is True.
    """
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
            refine_big=False,
            normalize_image=False,
            multi_gpu=False,
            fastmri=True,
            output_shape_spec=False,
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
        self.refine_big = refine_big
        if self.refine_big:
            n_layers_unet_sens = 4
            n_base_filters_unet_sens = 8
        else:
            n_layers_unet_sens = 3
            n_base_filters_unet_sens = 4
        self.normalize_image = normalize_image
        self.multi_gpu = multi_gpu
        self.output_shape_spec = output_shape_spec
        self._blocks_to_train = None
        if self.multi_gpu:
            self.available_gpus = get_gpus()
            self.n_gpus = len(self.available_gpus)
            if self.n_gpus > 1:
                self.n_iter = len(self.domain_sequence) // 2
            else:
                self.multi_gpu = False
        self.fastmri = fastmri
        if self.multicoil and self.refine_smaps:
            self.smaps_refiner = UnetComplex(
                n_layers=n_layers_unet_sens,
                layers_n_channels=[n_base_filters_unet_sens * 2**i for i in range(n_layers_unet_sens)],
                layers_n_non_lins=2,
                n_input_channels=1,
                n_output_channels=1,
                res=True,
                non_linearity='lrelu',
                channel_attention_kwargs=None,
                name=f'smaps_refiner',
            )

    @property
    def blocks_to_train(self):
        return self._blocks_to_train

    @blocks_to_train.setter
    def blocks_to_train(self, value):
        if isinstance(value, int):
            value = [value]
        self._blocks_to_train = value
        for i_domain, domain in enumerate(self.domain_sequence):
            trainable = self._blocks_to_train is None or i_domain // 2 in self._blocks_to_train
            if domain == 'K':
                try:
                    self.kspace_net[i_domain//2].trainable = trainable
                except AttributeError:
                    pass
            elif domain == 'I':
                self.image_net[i_domain//2].trainable = trainable


    def _refine_smaps(self, smaps):
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
        return smaps

    def k_domain_correction(self, i_domain, image_buffer, kspace_buffer, mask, smaps, original_kspace):
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
        return  kspace_buffer

    def i_domain_correction(self, i_domain, image_buffer, kspace_buffer, mask, smaps, *op_args):
        if self.i_buffer_mode:
            backward_op_res = self.backward_operator(kspace_buffer, mask, smaps, *op_args)
            if self.normalize_image:
                normalization_factor_iteration = tf.reduce_max(
                    tf.abs(backward_op_res),
                    axis=[1, 2, 3, 4] if self.multicoil else [1, 2, 3],
                    keepdims=True,
                )
                orig_bopres_shape = backward_op_res.shape
                normalization_factor_iteration = tf.cast(normalization_factor_iteration, image_buffer.dtype)
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
        return image_buffer


    def call(self, inputs):
        if self.multicoil:
            if self.output_shape_spec:
                # NOTE: for now we only consider the case of a specified output
                # shape when in multicoil
                if len(inputs) == 4:
                    original_kspace, mask, smaps, output_shape = inputs
                    op_args = ()
                else:
                    original_kspace, mask, smaps, output_shape, op_args = inputs
            elif len(inputs) == 3:
                original_kspace, mask, smaps = inputs
                output_shape = None
                op_args = ()
            else:
                original_kspace, mask, smaps, op_args = inputs
                output_shape = None
            if self.refine_smaps:
                smaps = self._refine_smaps(smaps)
        else:
            if len(inputs) == 2:
                original_kspace, mask = inputs
                op_args = ()
            else:
                original_kspace, mask, op_args = inputs
            smaps = None
            output_shape = None
        image = self.backward_operator(original_kspace, mask, smaps, *op_args)
        if self.normalize_image:
            normalization_factor = tf.reduce_max(
                tf.abs(image),
                axis=[1, 2, 3, 4] if self.multicoil else [1, 2, 3],
                keepdims=True,
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
            if not(self._blocks_to_train is None or i_domain // 2 <= max(self._blocks_to_train)):
                break
            with ExitStack() as stack:
                if self.multi_gpu:
                    i_gpu = gpu_index_from_submodel_index(
                        self.n_gpus,
                        self.n_iter,
                        i_domain//2,
                    )
                    stack.enter_context(tf.device(self.available_gpus[i_gpu]))
                if domain == 'K':
                    kspace_buffer = self.k_domain_correction(
                        i_domain,
                        image_buffer,
                        kspace_buffer,
                        mask,
                        smaps,
                        original_kspace,
                    )
                if domain == 'I':
                    image_buffer = self.i_domain_correction(
                        i_domain,
                        image_buffer,
                        kspace_buffer,
                        mask,
                        smaps,
                        *op_args,
                    )
        if self.normalize_image:
            image_buffer = image_buffer * normalization_factor
        if self.fastmri:
            image = general_fastmri_format(image_buffer[..., 0:1], output_shape)
        else:
            image = tf.abs(image_buffer[..., 0:1])
        image = tf.cast(image, tf.float32)
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

    def get_config(self):
        config = super(CrossDomainNet, self).get_config()
        config.update({
            'domain_sequence': self.domain_sequence,
            'data_consistency_mode': self.data_consistency_mode,
            'i_buffer_mode': self.i_buffer_mode,
            'k_buffer_mode': self.k_buffer_mode,
            'i_buffer_size': self.i_buffer_size,
            'k_buffer_size': self.k_buffer_size,
            'multicoil': self.multicoil,
            'refine_smaps': self.refine_smaps,
            'refine_big': self.refine_big,
            'normalize_image': self.normalize_image,
            'multi_gpu': self.multi_gpu,
            'fastmri': self.fastmri,
            'output_shape_spec': self.output_shape_spec,
        })
        return config
