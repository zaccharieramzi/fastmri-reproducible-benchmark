import glob
import random

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import Sequence

from ..helpers.fourier import FFT2
from ..helpers.utils import gen_mask


class Oasis2DSequence(Sequence):
    def __init__(self, path, mode='training', af=4):
        self.path = path
        self.mode = mode
        self.af = af

        self.filenames = glob.glob(path + '**/*.nii.gz')
        if not self.filenames:
            raise ValueError('No compressed nifti files at path {}'.format(path))
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
