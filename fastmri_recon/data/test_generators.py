import numpy as np
from skimage.draw import random_shapes

from ..helpers.fourier import fft
from ..helpers.reconstruction import zero_filled_recon
from ..helpers.utils import gen_mask
from ..helpers.threadsafe_gen import threadsafe_generator


class RandomShapeGenerator:
    def __init__(self, af, n_shapes, size, batch_size=1):
        self.af = af
        self.n_shapes = n_shapes
        self.size = size
        self.batch_size = batch_size
        self.im_shape = (batch_size, size, size, 1)


    @threadsafe_generator
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

    @threadsafe_generator
    def flow_z_filled_random_shapes(self,):
        random_shapes_gen = self.flow_random_shapes()
        for (kspaces, _), images in random_shapes_gen:
            z_filled = zero_filled_recon(kspaces[..., 0], crop=False)[..., None]
            yield z_filled, images

class DataGenerator:
    def __init__(self, af, data, batch_size=1):
        self.af = af
        self.data = data
        self.size = data.shape[1]
        self.batch_size = batch_size
        self.im_shape = (batch_size, data.shape[1], data.shape[1], 1)
        self.index = 0
        self.max_size = data.shape[0]


    @threadsafe_generator
    def flow_images(self,):
        while True:
            images = np.empty(self.im_shape)
            kspaces = np.empty(self.im_shape)
            if self.index == self.max_size:
                self.reset()
            for i in range(self.batch_size):
                image = self.data[self.index]
                images[i, ..., 0] = image

                image = image.astype('float32')
                image /= 255
                kspace = fft(image)
                kspaces[i, ..., 0] = kspace
                self.index += 1
            mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
            fourier_mask = np.repeat(mask.astype(np.float), self.size, axis=0)
            mask_batch = np.repeat(fourier_mask[None, ...], len(kspaces), axis=0)[..., None]
            kspaces *= mask_batch
            mask_batch = mask_batch[..., 0]
            yield (kspaces, mask_batch), images

    @threadsafe_generator
    def flow_z_filled_images(self,):
        random_shapes_gen = self.flow_images()
        for (kspaces, _), images in random_shapes_gen:
            z_filled = zero_filled_recon(kspaces[..., 0], crop=False)[..., None]
            yield z_filled, images

    def reset(self,):
        self.index = 0
