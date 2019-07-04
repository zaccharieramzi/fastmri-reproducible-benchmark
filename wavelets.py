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

    def op(self, data):
        if self.im_shape is None:
            self.im_shape = data.shape
            if not is_square(data):
                if np.argmax(data.shape) == 0:
                    self.big_dim = 0
                    self.diff = data.shape[0] - data.shape[1]
                    pad_seq = [(0, 0), (self.diff // 2, self.diff // 2)]
                else:
                    self.big_dim = 1
                    self.diff = data.shape[1] - data.shape[0]
                    pad_seq = [(self.diff // 2, self.diff // 2), (0, 0)]
                data = np.pad(data, pad_seq, mode='constant')
        else:
            if self.im_shape != data.shape:
                raise ValueError('Shape not corresponding')
        return super(WaveletDecimated, self).op(data)

    def adj_op(self, data):
        if self.im_shape is None:
            raise ValueError('Op not called first')
        im = super(WaveletDecimated, self).adj_op(data)
        if self.big_dim == 0:
            im = im[:, self.diff // 2: - self.diff // 2]
        if self.big_dim == 1:
            im = im[self.diff // 2: - self.diff // 2, :]
        return im
