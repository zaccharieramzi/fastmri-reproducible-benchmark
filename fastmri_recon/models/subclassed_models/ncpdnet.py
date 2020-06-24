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
            **kwargs,
        ):
        self.n_filters = n_filters
        self.n_primal = n_primal
        self.n_iter = n_iter
        self.activation = activation
        self.multicoil = multicoil
        self.im_size = im_size
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
        self.image_net = [CNNComplex(
            n_convs=3,
            n_filters=self.n_filters,
            n_output_channels=self.n_primal,
            activation='relu',
            res=True,
            name=f'image_net_{i}',
        ) for i in range(self.n_iter)]
        self.kspace_net = [measurements_residual for i in range(self.n_iter)]


def measurements_residual(concatenated_kspace):
    current_kspace = concatenated_kspace[..., 0:1]
    original_kspace = concatenated_kspace[..., 1:2]
    return current_kspace - original_kspace
