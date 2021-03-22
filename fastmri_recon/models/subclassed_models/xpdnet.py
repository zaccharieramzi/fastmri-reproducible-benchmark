from fastmri_recon.models.subclassed_models.denoisers.didn import DIDN
from .multiscale_complex import  MultiscaleComplex
from .cnn import  CNNComplex
from .cross_domain import CrossDomainNet
from ..utils.fourier import FFT, IFFT

class XPDNet(CrossDomainNet):
    r"""The XPDNet network, an extension of the Primal net.

    This extension of the networks presented in [A2017] allows to use image
    correction networks that are more complex than simply chained convolutions.
    It however doesn't feature (yet) the possibilit to have kspace corrections.

    It works with (potentially multicoil) masked cartesian Fourier transforms.

    Specifications of the image correction model:
        - input (tf.float32): nslices x h x w x 2*(n_primal + 1)
        - output (tf.float32): nslices x h x w x 2*n_primal

    Parameters:
        model_fun (function): the function initializing the image correction
            network. This allows to have different parameters for each block.
        model_kwargs (dict): the set of arguments used to initialize the image
            correction network.
        res (bool): whether we should add a residual connection for the image
            correction model. The residual connection will only take into account
            the first `n_primal` channel elements of the input.
            Defaults to False.
        n_scales (int): the number of scales the image correction network
            features. Defaults to 0, which means that no downsampling is
            performed in the image correction network.
        n_primal (int): the size of the buffer in the image space. Defaults to
            5.
        n_iter (int): the number of blocks for the unrolled reconstruction.
            Defaults to 10.
        multicoil (bool): whether the input data is multicoil. Defaults to False.
        **kwargs: keyword arguments for the CrossDomainNet.

    Attributes:
        same as CrossDomainNet.
    """
    def __init__(
            self,
            model_fun,
            model_kwargs,
            res=False,
            n_scales=0,
            n_primal=5,
            n_dual=1,
            n_dual_filters=16,
            n_iter=10,
            primal_only=True,
            multicoil=False,
            multiscale_kspace_learning=False,
            refine_smaps=False,
            **kwargs,
        ):
        self.model_fun = model_fun
        self.model_kwargs = model_kwargs
        self.res = res
        self.n_scales = n_scales
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.n_dual_filters = n_dual_filters
        self.n_iter = n_iter
        self.primal_only = primal_only
        self.multicoil = multicoil
        self.multiscale_kspace_learning = multiscale_kspace_learning
        self.refine_smaps = refine_smaps
        super(XPDNet, self).__init__(
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
        self.image_net = [MultiscaleComplex(
            model_fun=self.model_fun,
            model_kwargs=self.model_kwargs,
            res=self.res,
            n_output_channels=self.n_primal,
            n_scales=self.n_scales,
            name=f'image_net_{i}',
        ) for i in range(self.n_iter)]
        if not self.primal_only:
            if self.multiscale_kspace_learning:
                self.kspace_net = [MultiscaleComplex(
                    model_fun=DIDN,
                    model_kwargs=dict(
                        # rather small didn
                        n_filters=32,
                        n_dubs=2,
                        n_convs_recon=3,
                        n_scales=3,
                        n_outputs=2*self.n_dual,
                    ),
                    res=True,
                    n_output_channels=self.n_dual,
                    n_scales=3,
                    multicoil=True,
                    name=f'kspace_net_{i}',
                ) for i in range(self.n_iter)]
            else:
                self.kspace_net = [CNNComplex(
                    n_convs=3,
                    n_filters=self.n_dual_filters,
                    n_output_channels=self.n_dual,
                    activation='relu',
                    res=True,
                    multicoil=self.multicoil,
                    name=f'kspace_net_{i}',
                ) for i in range(self.n_iter)]
        else:
            # TODO: check n dual
            # TODO: code small diff function
            self.kspace_net = [measurements_residual for i in range(self.n_iter)]

    def get_config(self):
        config = super(XPDNet, self).get_config()
        config.update({
            'model_fun': self.model_fun,
            'model_kwargs': self.model_kwargs,
            'res': self.res,
            'n_scales': self.n_scales,
            'n_primal': self.n_primal,
            'n_dual': self.n_dual,
            'n_dual_filters': self.n_dual_filters,
            'n_iter': self.n_iter,
            'primal_only': self.primal_only,
            'multicoil': self.multicoil,
            'multiscale_kspace_learning': self.multiscale_kspace_learning,
            'refine_smaps': self.refine_smaps,
        })


def measurements_residual(concatenated_kspace):
    current_kspace = concatenated_kspace[..., 0:1]
    original_kspace = concatenated_kspace[..., 1:2]
    return current_kspace - original_kspace
