import tensorflow as tf
from tensorflow.keras.models import Model

from fastmri_recon.models.utils.fourier import NFFT, AdjNFFT
from fastmri_recon.models.utils.fastmri_format import general_fastmri_format

class NCDcompReconstructor(Model):
    def __init__(self, im_size=(640, 474), multicoil=False, fastmri_format=True, brain=False, **kwargs):
        self.im_size = im_size
        self.multicoil = multicoil
        super(NCDcompReconstructor, self).__init__(**kwargs)
        self.adj_op = AdjNFFT(im_size=self.im_size, multicoil=self.multicoil, density_compensation=True)
        self.fastmri_format = fastmri_format
        self.brain = brain

    def call(self, inputs):
        if self.brain:
            image = self.adj_op([*inputs[:-2], *inputs[-1]])
            output_shape = inputs[-2]
        else:
            image = self.adj_op([*inputs[:-1], *inputs[-1]])
            output_shape = None
        if self.fastmri_format:
            image = general_fastmri_format(image, output_shape=output_shape)
        else:
            image = tf.abs(image)
        return image
