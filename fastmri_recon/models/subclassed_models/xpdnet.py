from .multiscale_complex import  MultiscaleComplex
from .cross_domain import CrossDomainNet
from ..utils.fourier import FFT, IFFT

class XPDNet(CrossDomainNet):
    def __init__(
            self,
            model,
            res=False,
            n_scales=0,
            n_primal=5,
            n_iter=10,
            multicoil=False,
            refine_smaps=False,
            **kwargs,
        ):
        self.res = res
        self.n_scales = n_scales
        self.n_primal = n_primal
        self.n_iter = n_iter
        self.multicoil = multicoil
        self.refine_smaps = refine_smaps
        super(XPDNet, self).__init__(
            domain_sequence='KI'*self.n_iter,
            data_consistency_mode='measurements_residual',
            i_buffer_mode=True,
            k_buffer_mode=False,
            i_buffer_size=self.n_primal,
            k_buffer_size=1,
            multicoil=self.multicoil,
            refine_smaps=self.refine_smaps,
            **kwargs,
        )
        self.model = model
        self.op = FFT(masked=True, multicoil=self.multicoil)
        self.adj_op = IFFT(masked=True, multicoil=self.multicoil)
        self.image_net = [MultiscaleComplex(
            model=self.model,
            res=self.res,
            n_output_channels=self.n_primal,
            n_scales=self.n_scales,
            name=f'image_net_{i}',
        ) for i in range(self.n_iter)]
        self.kspace_net = [measurements_residual for i in range(self.n_iter)]


def measurements_residual(concatenated_kspace):
    current_kspace = concatenated_kspace[..., 0:1]
    original_kspace = concatenated_kspace[..., 1:2]
    return current_kspace - original_kspace
