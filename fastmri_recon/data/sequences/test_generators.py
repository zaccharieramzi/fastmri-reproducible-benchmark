import numpy as np
from skimage.draw import random_shapes

from ...evaluate.reconstruction.zero_filled_reconstruction import zero_filled_recon
from ..utils.masking.gen_mask import gen_mask
from ..utils.fourier import fft


class RandomShapeGenerator:
    def __init__(self, af, n_shapes, size, batch_size=1):
        self.af = af
        self.n_shapes = n_shapes
        self.size = size
        self.batch_size = batch_size
        self.im_shape = (batch_size, size, size, 1)

    def flow_random_shapes(self,):
        while True:
            images = np.empty(self.im_shape)
            kspaces = np.empty(self.im_shape)
            for i in range(self.batch_size):
                image, _ = random_shapes(
                    (self.size, self.size),
                    max_shapes=self.n_shapes,
                    multichannel=False,
                    allow_overlap=True,
                    max_size=self.size,
                )
                image = image.astype('float32')
                image /= 255
                kspace = fft(image)
                images[i, ..., 0] = image
                kspaces[i, ..., 0] = kspace
            mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
            fourier_mask = np.repeat(mask.astype(np.float), self.size, axis=0)
            mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)[..., None]
            kspaces *= mask_batch
            mask_batch = mask_batch[..., 0]
            yield (kspaces, mask_batch), images

    def flow_z_filled_random_shapes(self,):
        random_shapes_gen = self.flow_random_shapes()
        for (kspaces, _), images in random_shapes_gen:
            z_filled = zero_filled_recon(kspaces[..., 0], crop=False)[..., None]
            yield z_filled, images
