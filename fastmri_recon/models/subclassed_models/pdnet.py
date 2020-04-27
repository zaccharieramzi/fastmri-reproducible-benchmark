from .cnn import  CNNComplex
from .cross_domain import CrossDomainNet
from ..utils.fourier import FFT, IFFT

class PDNet(CrossDomainNet):
    def __init__(
            self,
            n_filters=32,
            n_primal=5,
            n_dual=5,
            n_iter=10,
            primal_only=False,
            activation='relu',
            multicoil=False,
            **kwargs,
        ):
        self.n_filters = n_filters
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.n_iter = n_iter
        self.primal_only = primal_only
        self.activation = activation
        self.multicoil = multicoil
        super(PDNet, self).__init__(
            domain_sequence='KI'*self.n_iter,
            data_consistency_mode='measurements_residual',
            i_buffer_mode=True,
            k_buffer_mode=not self.primal_only,
            i_buffer_size=self.n_primal,
            k_buffer_size=self.n_dual,
            multicoil=self.multicoil,
            **kwargs,
        )
        self.op = FFT(masked=True, multicoil=self.multicoil)
        self.adj_op = IFFT(masked=True, multicoil=self.multicoil)
        self.image_net = [CNNComplex(
            n_convs=3,
            n_filters=self.n_filters,
            n_output_channels=self.n_primal,
            activation='relu',
            res=True,
            name=f'image_net_{i}',
        ) for i in range(self.n_iter)]
        if not self.primal_only:
            # TODO: check that when multicoil we do not have this
            self.kspace_net = [CNNComplex(
                n_convs=3,
                n_filters=self.n_filters,
                n_output_channels=self.n_dual,
                activation='relu',
                res=True,
                name=f'kspace_net_{i}',
            ) for i in range(self.n_iter)]
        else:
            # TODO: check n dual
            # TODO: code small diff function
            self.kspace_net = [measurements_residual for i in range(self.n_iter)]


def measurements_residual(concatenated_kspace):
    current_kspace = concatenated_kspace[..., 0:1]
    original_kspace = concatenated_kspace[..., 1:2]
    return current_kspace - original_kspace
