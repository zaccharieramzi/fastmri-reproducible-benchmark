from .unet import  UnetComplex
from .cross_domain import CrossDomainNet
from ..utils.fourier import FFT, IFFT

class UPDNet(CrossDomainNet):
    def __init__(
            self,
            n_layers=3,
            layers_n_channels=[8, 16, 32],
            res=True,
            non_linearity='relu',
            channel_attention_kwargs=None,
            n_primal=5,
            n_dual=5,
            n_iter=10,
            primal_only=False,
            multicoil=False,
            refine_smaps=False,
            **kwargs,
        ):
        self.n_layers = n_layers
        self.layers_n_channels = layers_n_channels
        self.res = res
        self.non_linearity = non_linearity
        self.channel_attention_kwargs = channel_attention_kwargs
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.n_iter = n_iter
        self.primal_only = primal_only
        self.multicoil = multicoil
        self.refine_smaps = refine_smaps
        super(UPDNet, self).__init__(
            domain_sequence='KI'*self.n_iter,
            data_consistency_mode='measurements_residual',
            i_buffer_mode=True,
            k_buffer_mode=not self.primal_only,
            i_buffer_size=self.n_primal,
            k_buffer_size=self.n_dual,
            multicoil=self.multicoil,
            refine_smaps=self.refine_smaps,
            **kwargs,
        )
        self.op = FFT(masked=True, multicoil=self.multicoil)
        self.adj_op = IFFT(masked=True, multicoil=self.multicoil)
        self.image_net = [UnetComplex(
            n_layers=self.n_layers,
            layers_n_channels=self.layers_n_channels,
            layers_n_non_lins=2,
            n_input_channels=self.n_primal + 1,
            n_output_channels=self.n_primal,
            res=self.res,
            non_linearity=self.non_linearity,
            channel_attention_kwargs=channel_attention_kwargs,
            name=f'image_net_{i}',
        ) for i in range(self.n_iter)]
        if not self.primal_only:
            # TODO: check that when multicoil we do not have this
            self.kspace_net = [UnetComplex(
                n_layers=self.n_layers,
                layers_n_channels=self.layers_n_channels,
                layers_n_non_lins=2,
                n_output_channels=self.n_dual,
                n_input_channels=self.n_dual + 2,
                res=self.res,
                non_linearity=self.non_linearity,
                channel_attention_kwargs=channel_attention_kwargs,
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
