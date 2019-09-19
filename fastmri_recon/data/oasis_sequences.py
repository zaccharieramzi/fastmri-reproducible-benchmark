import glob
import os.path as op
import random
import re

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import Sequence

from ..helpers.fourier import FFT2
from ..helpers.utils import gen_mask


def get_session_from_filename(filename):
    base_name = op.basename(filename)
    session_id = re.findall(r'ses-d\d{4}', base_name)[0]
    return session_id

class Oasis2DSequence(Sequence):
    def __init__(self, path, mode='training', af=4, val_split=0.1, filenames=None, seed=None):
        self.path = path
        self.mode = mode
        self.af = af

        if filenames is None:
            self.filenames = glob.glob(path + '**/*.nii.gz', recursive=True)
            if not self.filenames:
                raise ValueError('No compressed nifti files at path {}'.format(path))
            if val_split > 0:
                sessions = [get_session_from_filename(filename) for filename in self.filenames]
                n_val = int(len(sessions) * val_split)
                random.seed(seed)
                random.shuffle(sessions)
                val_sessions = sessions[:n_val]
                val_filenames = [filename for filename in self.filenames if get_session_from_filename(filename) in val_sessions]
                self.filenames = [filename for filename in self.filenames if filename not in val_filenames]
                self.val_sequence = type(self)(path, mode=mode, af=af, val_split=0, filenames=val_filenames)
            else:
                self.val_sequence = None
        else:
            self.filenames = filenames
            self.val_sequence = None
        self.filenames.sort()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        images = nib.load(filename)
        images = images.get_fdata()
        images = images[..., None]
        images = images.astype(np.complex64)
        return images


class Masked2DSequence(Oasis2DSequence):
    def __init__(self, *args, inner_slices=None, rand=False, scale_factor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_slices = inner_slices
        self.rand = rand
        self.scale_factor = scale_factor
        if self.val_sequence is not None:
            self.val_sequence.inner_slices = inner_slices
            self.val_sequence.scale_factor = scale_factor

    def __getitem__(self, idx):
        """Get a training triplet from the file at `idx` in `self.filenames`.

        This method will get the images at filename,select only the relevant
        slices, create a mask on-the-fly, mask the kspaces (obtained from
        the images) with it, and return a tuple ((kspaces, mask), images).

        Parameters:
        filename (str): index of the nii.gz file containing the training data
            in `self.filenames`.

        Returns:
        tuple ((ndarray, ndarray), ndarray): the masked kspaces, mask and images
        corresponding to the volume in NHWC format (mask is NHW).
        """
        images = super(Masked2DSequence, self).__getitem__(idx)
        if self.inner_slices is not None:
            n_slices = len(images)
            slice_start = n_slices // 2 - self.inner_slices // 2
            if self.rand:
                i_slice = random.randint(slice_start, slice_start + self.inner_slices)
                selected_slices = slice(i_slice, i_slice + 1)
            else:
                selected_slices = slice(slice_start, slice_start + self.inner_slices)
        images = images[selected_slices]
        k_shape = images[0].shape
        kspaces = np.empty_like(images)
        mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        fourier_op = FFT2(fourier_mask)
        for i, image in enumerate(images):
            kspaces[i] = fourier_op.op(image[..., 0])[..., None]
        mask_batch = mask[None, ...]
        scale_factor = self.scale_factor
        kspaces_scaled = kspaces * scale_factor
        images_scaled = images * scale_factor
        return ([kspaces_scaled, mask_batch], images_scaled)
