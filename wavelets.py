import numpy as np

from mri.reconstruct.linear import Wavelet2


def is_square(data):
    return data.shape[0] == data.shape[1]

class WaveletDecimated(Wavelet2):

    def __init__(self, *args, **kwargs):
        super(WaveletDecimated, self).__init__(*args, **kwargs)
        self.im_shape = None
        self.big_dim = None
        self.diff = None
        self.pad_seq = None

    def pad_image(self, image):
        image = np.pad(image, self.pad_seq, mode='constant')
        return image

    def op(self, data):
        if self.im_shape is None:
            self.im_shape = data.shape
            if not is_square(data):
                if np.argmax(data.shape) == 0:
                    self.big_dim = 0
                    self.diff = data.shape[0] - data.shape[1]
                    self.pad_seq = [(0, 0), (self.diff // 2, self.diff // 2)]
                else:
                    self.big_dim = 1
                    self.diff = data.shape[1] - data.shape[0]
                    self.pad_seq = [(self.diff // 2, self.diff // 2), (0, 0)]
        else:
            if self.im_shape != data.shape:
                raise ValueError('Shape not corresponding')
        if not is_square(data):
            data = self.pad_image(data)
        # import pdb; pdb.set_trace()
        return super(WaveletDecimated, self).op(data)

    def adj_op(self, data):
        if self.im_shape is None:
            raise ValueError('Op not called first')
        im = super(WaveletDecimated, self).adj_op(data)
        if self.big_dim is not None:
            if self.big_dim == 0:
                im = im[:, self.diff // 2: - self.diff // 2]
            if self.big_dim == 1:
                im = im[self.diff // 2: - self.diff // 2, :]
        return im
