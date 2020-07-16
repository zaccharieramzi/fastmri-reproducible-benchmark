import tensorflow as tf

from .cnn import  CNNComplex
from .cross_domain import CrossDomainNet
from ..utils.fourier import NFFT, AdjNFFT

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
            **kwargs,
        ):
        self.n_filters = n_filters
        self.n_primal = n_primal
        self.n_iter = n_iter
        self.activation = activation
        self.multicoil = multicoil
        self.im_size = im_size
        self.three_d = three_d
        super(NCPDNet, self).__init__(
            domain_sequence='KI'*self.n_iter,
            data_consistency_mode='measurements_residual',
            i_buffer_mode=True,
            k_buffer_mode=False,
            i_buffer_size=self.n_primal,
            k_buffer_size=1,
            multicoil=self.multicoil,
            **kwargs,
        )
        self.op = NFFT(im_size=self.im_size, multicoil=self.multicoil)
        self.adj_op = AdjNFFT(im_size=self.im_size, multicoil=self.multicoil, density_compensation=dcomp)
        available_gpus = [
            d for d in tf.config.list_physical_devices()
            if d.device_type == 'GPU'
        ]
        n_gpus = len(available_gpus)
        if n_gpus:
            self.image_net = []
            for i in range(self.n_iter):
                i_gpu = i
                with tf.device(available_gpus[i_gpu]):
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
        else:
            self.image_net = [CNNComplex(
                n_convs=3,
                n_filters=self.n_filters,
                n_output_channels=self.n_primal,
                activation='relu',
                res=True,
                three_d=self.three_d,
                name=f'image_net_{i}',
            ) for i in range(self.n_iter)]
        self.kspace_net = [measurements_residual for i in range(self.n_iter)]


def measurements_residual(concatenated_kspace):
    current_kspace = concatenated_kspace[..., 0:1]
    original_kspace = concatenated_kspace[..., 1:2]
    return current_kspace - original_kspace
