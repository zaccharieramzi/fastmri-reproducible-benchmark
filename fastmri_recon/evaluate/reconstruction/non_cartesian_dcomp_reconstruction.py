from tensorflow.keras.models import Model

from fastmri_recon.models.utils.fourier import NFFT, AdjNFFT


class NCDcompReconstructor(Model):
    def __init__(self, im_size=(640, 474), multicoil=False, **kwargs):
        self.im_size = im_size
        self.multicoil = multicoil
        super(NCDcompReconstructor, self).__init__(**kwargs)
        self.adj_op = AdjNFFT(im_size=self.im_size, multicoil=self.multicoil, density_compensation=True)

    def call(self, inputs):
        if self.multicoil:
            raise NotImplementedError('Multicoil has yet to be implemented')
        else:
            original_kspace, ktraj, op_args = inputs
        image = self.adj_op([original_kspace, ktraj, *op_args])
        image = tf_fastmri_format(image)
        return image
