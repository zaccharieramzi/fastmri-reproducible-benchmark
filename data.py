import glob
import time

import h5py
from keras.utils import Sequence
import numpy as np

from fourier import FFT2
from utils import crop_center, gen_mask

def from_test_file_to_mask_and_kspace(filename):
    try:
        with  h5py.File(filename) as h5_obj:
            masks = h5_obj['mask'][()]
            kspaces = h5_obj['kspace'][()]
            return masks, kspaces
    except OSError:
        time.sleep(0.3)
        return from_test_file_to_mask_and_kspace(filename)


def from_train_file_to_image_and_kspace(filename):
    try:
        with h5py.File(filename) as h5_obj:
            images = h5_obj['reconstruction_esc'][()]
            kspaces = h5_obj['kspace'][()]
            return images, kspaces
    except OSError:
        time.sleep(1)
        return from_train_file_to_image_and_kspace(filename)

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
    n_samples_per_mode = {
        'training': 34742,
        'validation': 7135,
        'testing': 3903,
    }
    train_modes = ('training', 'validation')

    def __init__(self, path, mode='training', batch_size=32, af=4):
        self.path = path
        self.mode = mode
        self.batch_size = batch_size
        self.af = af

        filenames = glob.glob(path + '*.h5')
        if not filenames:
            raise ValueError('No h5 files at path {}'.format(path))
        filenames.sort()
        self.idx_to_file = list()
        for filename in filenames:
            kspaces = from_file_to_kspace(filename)
            self.idx_to_file += [(filename, i) for i in range(len(kspaces))]


    def __len__(self):
        """From fastMRI paper"""
        return type(self).n_samples_per_mode[self.mode]

    def __getitem__(self, idx):
        filenames_and_positions = self.idx_to_file[idx:idx + self.batch_size]
        if self.mode in type(self).train_modes:
            return self.get_item_train(filenames_and_positions)
        else:
            return self.get_item_test(filenames_and_positions)


    def get_item_train(self, filenames_and_positions):
        last_filename = None
        current_batch = list()
        current_batch_zero = list()
        for filename, i_kspace in filenames_and_positions:
            if last_filename != filename:
                images, kspaces = from_train_file_to_image_and_kspace(filename)
                mask = gen_mask(kspaces[0], accel_factor=self.af)
                fourier_mask = np.repeat(mask.astype(np.float)[None, :], kspaces[0].shape[0], axis=0)
            zero_filled_rec = zero_filled(kspaces[i_kspace] * fourier_mask)
            zero_filled_rec = zero_filled_rec[:, :, None]
            image = images[i_kspace][:, :, None]
            current_batch_zero.append(zero_filled_rec)
            current_batch.append(image)
        zero_img_batch = np.array(current_batch_zero)
        img_batch = np.array(current_batch)
        return (zero_img_batch, img_batch)


    def get_item_test(self, filenames_and_positions):
        last_filename = None
        current_batch_zero = list()
        for filename, i_kspace in filenames_and_positions:
            if last_filename != filename:
                mask, kspaces = from_test_file_to_mask_and_kspace(filename)
                mask_af = len(mask) / sum(mask)
                if not(self.af == 4 and mask_af < 5.5 or self.af == 8 and mask_af > 8):
                    # TODO: I think this won't work: I will probably need to check the filenames first
                    continue
            zero_filled_rec = zero_filled(kspaces[i_kspace])
            zero_filled_rec = zero_filled_rec[:, :, None]
            current_batch_zero.append(zero_filled_rec)
        zero_img_batch = np.array(current_batch_zero)
        return zero_img_batch



def zero_filled_2d_generator(path, mode='training', batch_size=32, af=4):
    train_modes = ('training', 'validation')
    filenames = glob.glob(path + '*.h5')
    if not filenames:
        raise ValueError('No h5 files at path {}'.format(path))
    while True:
        current_batch_zero = []
        current_batch = []
        i_slice = 0
        for filename in filenames:
            if mode in train_modes:
                try:
                    images, kspaces = from_train_file_to_image_and_kspace(filename)
                except OSError:
                    continue
                mask = gen_mask(kspaces[0], accel_factor=af)
                fourier_mask = np.repeat(mask.astype(np.float)[None, :], kspaces[0].shape[0], axis=0)
                for image, kspace in zip(images, kspaces):
                    i_slice += 1
                    zero_filled_rec = zero_filled(kspace * fourier_mask)
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
                try:
                    mask, kspaces = from_test_file_to_mask_and_kspace(filename)
                except OSError:
                    continue
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
                try:
                    images, kspaces = from_train_file_to_image_and_kspace(filename)
                except OSError:
                    continue
                mask = gen_mask(kspaces[0], accel_factor=af)
                fourier_mask = np.repeat(mask.astype(np.float)[None, :], kspaces[0].shape[0], axis=0)
                zero_filled_recs = list()
                for kspace in kspaces:
                    zero_filled_rec = zero_filled(fourier_mask * kspace)
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
                try:
                    mask, kspaces = from_test_file_to_mask_and_kspace(filename)
                except OSError:
                    continue
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
