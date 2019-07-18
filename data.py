import glob
import time

import h5py
from keras.utils import Sequence
import numpy as np

from fourier import FFT2
from utils import crop_center, gen_mask

def from_test_file_to_mask_and_kspace(filename):
    with  h5py.File(filename) as h5_obj:
        masks = h5_obj['mask'][()]
        kspaces = h5_obj['kspace'][()]
        return masks, kspaces



def from_train_file_to_image_and_kspace(filename):
    with h5py.File(filename) as h5_obj:
        images = h5_obj['reconstruction_esc'][()]
        kspaces = h5_obj['kspace'][()]
        return images, kspaces


def from_file_to_kspace(filename):
    with h5py.File(filename) as h5_obj:
        kspaces = h5_obj['kspace'][()]
        return kspaces


def zero_filled(kspace):
    fourier_op = FFT2(np.ones_like(kspace))
    im_recon = np.abs(fourier_op.adj_op(kspace))
    im_cropped = crop_center(im_recon, 320)
    return im_cropped


class ZeroFilled2DSequence(Sequence):
    train_modes = ('training', 'validation')

    def __init__(self, path, mode='training', af=4):
        self.path = path
        self.mode = mode
        self.af = af

        self.filenames = glob.glob(path + '*.h5')
        if not self.filenames:
            raise ValueError('No h5 files at path {}'.format(path))
        self.filenames.sort()
        if mode == 'testing':
            af_filenames = list()
            for filename in self.filenames:
                mask, _ = from_test_file_to_mask_and_kspace(filename)
                mask_af = len(mask) / sum(mask)
                if af == 4 and mask_af < 5.5 or af == 8 and mask_af > 8:
                    af_filenames.append(filename)
            self.filenames = af_filenames


    def __len__(self):
        """From fastMRI paper"""
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        if self.mode in type(self).train_modes:
            return self.get_item_train(filename)
        else:
            return self.get_item_test(filename)


    def get_item_train(self, filename):
        images, kspaces = from_train_file_to_image_and_kspace(filename)
        mask = gen_mask(kspaces[0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), kspaces[0].shape[0], axis=0)
        zero_img_batch = list()
        for kspace in kspaces:
            zero_filled_rec = zero_filled(kspace * fourier_mask)
            zero_filled_rec = zero_filled_rec[:, :, None]
            zero_img_batch.append(zero_filled_rec)
        zero_img_batch = np.array(zero_img_batch)
        images = images[..., None]
        return (zero_img_batch, images)


    def get_item_test(self, filename):
        _, kspaces = from_test_file_to_mask_and_kspace(filename)
        zero_img_batch = list()
        for kspace in kspaces:
            zero_filled_rec = zero_filled(kspace)
            zero_filled_rec = zero_filled_rec[:, :, None]
            zero_img_batch.append(zero_filled_rec)
        zero_img_batch = np.array(zero_img_batch)
        return zero_img_batch
