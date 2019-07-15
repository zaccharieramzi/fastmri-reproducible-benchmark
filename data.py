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
        mask = h5_obj['mask'][()]
        return images, kspaces, mask


def zero_filled(kspace):
    fourier_op = FFT2(np.ones_like(kspace))
    im_recon = np.abs(fourier_op.adj_op(kspace))
    im_cropped = crop_center(im_recon, 320)
    return im_cropped


def zero_filled_2d_generator(path, mode='training', batch_size=32, af=None):
    train_modes = ('training', 'validation')
    filenames = glob.glob(path + '*.h5')
    while True:
        current_batch_zero = []
        current_batch = []
        i_slice = 0
        for filename in filenames:
            if mode in train_modes:
                images, kspaces, mask = from_train_file_to_image_and_kspace(filename)
                if af is not None:
                    mask_af = len(mask) / sum(mask)
                    if not(af == 4 and mask_af < 5.5 or af == 8 and mask_af > 8):
                        continue
                for image, kspace in zip(images, kspaces):
                    i_slice += 1
                    zero_filled_rec = zero_filled(kspace)
                    zero_filled_rec = zero_filled_rec[:, :, None]
                    image = image[:, :, None]
                    current_batch_zero.append(zero_filled_rec)
                    current_batch.append(image)
                    if i_slice % batch_size == 0:
                        zero_img_batch = np.array(current_batch_zero)
                        img_batch = np.array(current_batch)
                        current_batch = []
                        current_batch_zero = []
                        yield (zero_img_batch, img_batch)
            else:
                mask, kspaces = from_test_file_to_mask_and_kspace(filename)
                if af is not None:
                    mask_af = len(mask) / sum(mask)
                    if not(af == 4 and mask_af < 5.5 or af == 8 and mask_af > 8):
                        continue
                for kspace in kspaces:
                    i_slice += 1
                    zero_filled_rec = zero_filled(kspace)
                    zero_filled_rec = zero_filled_rec[:, :, None]
                    current_batch_zero.append(zero_filled_rec)
                    if i_slice % batch_size == 0:
                        zero_img_batch = np.array(current_batch_zero)
                        current_batch_zero = []
                        yield zero_img_batch


def zero_filled_3d_generator(path, mode='training', batch_size=32, af=None):
    train_modes = ('training', 'validation')
    filenames = glob.glob(path + '*.h5')
    while True:
        current_batch_zero = []
        current_batch = []
        for i_file, filename in enumerate(filenames):
            if mode in train_modes:
                images, kspaces, mask = from_train_file_to_image_and_kspace(filename)
                if af is not None:
                    mask_af = len(mask) / sum(mask)
                    if not(af == 4 and mask_af < 5.5 or af == 8 and mask_af > 8):
                        continue
                zero_filled_recs = list()
                for kspace in kspaces:
                    zero_filled_rec = zero_filled(kspace)
                    zero_filled_recs.append(zero_filled_rec)
                zero_filled_recs = np.array(zero_filled_recs)
                zero_filled_recs = zero_filled_recs[..., None]
                images = images[..., None]
                current_batch_zero.append(zero_filled_recs)
                current_batch.append(images)
                if (i_file + 1) % batch_size == 0:
                    zero_img_batch = np.array(current_batch_zero)
                    img_batch = np.array(current_batch)
                    current_batch = []
                    current_batch_zero = []
                    yield (zero_img_batch, img_batch)
            else:
                mask, kspaces = from_test_file_to_mask_and_kspace(filename)
                if af is not None:
                    mask_af = len(mask) / sum(mask)
                    if not(af == 4 and mask_af < 5.5 or af == 8 and mask_af > 8):
                        continue
                zero_filled_recs = list()
                for kspace in kspaces:
                    zero_filled_rec = zero_filled(kspace)
                    zero_filled_recs.append(zero_filled_rec)
                zero_filled_recs = np.array(zero_filled_recs)
                zero_filled_recs = zero_filled_recs[..., None]
                current_batch_zero.append(zero_filled_recs)
                if (i_file + 1) % batch_size == 0:
                    zero_img_batch = np.array(current_batch_zero)
                    current_batch_zero = []
                    yield zero_img_batch
