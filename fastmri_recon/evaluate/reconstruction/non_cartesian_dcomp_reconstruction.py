from tensorflow.keras.models import Model

from fastmri_recon.models.utils.fourier import NFFT, AdjNFFT
from fastmri_recon.models.utils.fastmri_format import tf_fastmri_format

class NCDcompReconstructor(Model):
    def __init__(self, im_size=(640, 474), multicoil=False, fastmri_format=True, **kwargs):
        self.im_size = im_size
        self.multicoil = multicoil
        super(NCDcompReconstructor, self).__init__(**kwargs)
        self.adj_op = AdjNFFT(im_size=self.im_size, multicoil=self.multicoil, density_compensation=True)
        self.fastmri_format = fastmri_format

    def call(self, inputs):
        image = self.adj_op([*inputs[:-1], *inputs[-1]])
        if self.fastmri_format:
            image = tf_fastmri_format(image)
        else:
            image = tf.abs(image)
        return image
