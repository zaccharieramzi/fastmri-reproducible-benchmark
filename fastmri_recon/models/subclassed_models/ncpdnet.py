from contextlib import ExitStack

import tensorflow as tf

from .cnn import  CNNComplex
from .cross_domain import CrossDomainNet
from ..utils.fourier import NFFT, AdjNFFT
from ..utils.gpu_placement import gpu_index_from_submodel_index, get_gpus

class NCPDNet(CrossDomainNet):
    def __init__(
            self,
            n_filters=32,
            n_primal=5,
            n_iter=10,
            activation='relu',
            multicoil=False,
            dcomp=False,
            im_size=(640, 474),
            three_d=False,
            grad_traj=False,
            nufft_implementation='tfkbnufft',
            **kwargs,
        ):
        self.n_filters = n_filters
        self.n_primal = n_primal
        self.n_iter = n_iter
        self.activation = activation
        self.multicoil = multicoil
        self.im_size = im_size
        self.three_d = three_d
        self.grad_traj = grad_traj
        super(NCPDNet, self).__init__(
            domain_sequence='KI'*self.n_iter,
            data_consistency_mode='measurements_residual',
            i_buffer_mode=True,
            k_buffer_mode=False,
            i_buffer_size=self.n_primal,
            k_buffer_size=1,
            multicoil=self.multicoil,
            multi_gpu=True,
            **kwargs,
        )
        self.op = NFFT(
            im_size=self.im_size,
            multicoil=self.multicoil,
            grad_traj=self.grad_traj,
            implementation=nufft_implementation
        )
        self.adj_op = AdjNFFT(
            im_size=self.im_size,
            multicoil=self.multicoil,
            density_compensation=dcomp,
            grad_traj=self.grad_traj,
            implementation=nufft_implementation
        )
        available_gpus = get_gpus()
        n_gpus = len(available_gpus)
        self.image_net = []
        for i in range(self.n_iter):
            with ExitStack() as stack:
                if n_gpus > 1:
                    i_gpu = gpu_index_from_submodel_index(n_gpus, self.n_iter, i)
                    stack.enter_context(tf.device(available_gpus[i_gpu]))
                image_model = CNNComplex(
                    n_convs=3,
                    n_filters=self.n_filters,
                    n_output_channels=self.n_primal,
                    activation='relu',
                    res=True,
                    three_d=self.three_d,
                    name=f'image_net_{i}',
                )
            self.image_net.append(image_model)
        self.kspace_net = [measurements_residual for i in range(self.n_iter)]


def measurements_residual(concatenated_kspace):
    current_kspace = concatenated_kspace[..., 0:1]
    original_kspace = concatenated_kspace[..., 1:2]
    return current_kspace - original_kspace
