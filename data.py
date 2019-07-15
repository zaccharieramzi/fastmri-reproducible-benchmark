import glob

import h5py
import numpy as np

from fourier import FFT2
from utils import crop_center

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


def zero_filled(kspace):
    fourier_op = FFT2(np.ones_like(kspace))
    im_recon = np.abs(fourier_op.adj_op(kspace))
    im_cropped = crop_center(im_recon, 320)
    return im_cropped


def zero_filled_2d_generator(path, mode='training', batch_size=32):
    train_modes = ('training', 'validation')
    filenames = glob.glob(path + '*.h5')
    while True:
        current_batch_zero = []
        current_batch = []
        i_slice = 0
        for filename in filenames:
            if mode in train_modes:
                images, kspaces = from_train_file_to_image_and_kspace(filename)
                for image, kspace in zip(images, kspaces):
                    i_slice += 1
                    zero_filled_rec = zero_filled(kspace)
                    zero_filled_rec = zero_filled_rec[:, :, None]
                    current_batch_zero.append(zero_filled_rec)
                    current_batch.append(image)
                    if i_slice % batch_size == 0:
                        zero_img_batch = np.array(current_batch_zero)
                        img_batch = np.array(current_batch)
                        current_batch = []
                        current_batch_zero = []
                        yield (zero_img_batch, img_batch)
            else:
                _, kspaces = from_test_file_to_mask_and_kspace(filename)
                for kspace in kspaces:
                    i_slice += 1
                    zero_filled_rec = zero_filled(kspace)
                    zero_filled_rec = zero_filled_rec[:, :, None]
                    current_batch_zero.append(zero_filled_rec)
                    if i_slice % batch_size == 0:
                        zero_img_batch = np.array(current_batch_zero)
                        current_batch_zero = []
                        yield zero_img_batch
