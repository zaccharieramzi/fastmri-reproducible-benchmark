import glob
import os.path as op
import random
import re

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import Sequence

from ..helpers.fourier import FFT2
from ..helpers.reconstruction import zero_filled_recon
from ..helpers.utils import gen_mask


def _get_session_from_filename(filename):
    base_name = op.basename(filename)
    session_id = re.findall(r'ses-d\d{4}', base_name)[0]
    return session_id

class Oasis2DSequence(Sequence):
    """The base class for using the OASIS data in keras.
    You need to specify the path to the type of data you want, the mode of
    the sequence, its acceleration rate and the validation split.
    This will by default enumerate volumes.

    Parameters:
    path (str): the path to the data of this sequence. The data must be in
    nii.gz files.
    mode (str): the mode of sequence in ['training', 'validation'].
    The mode training is to be used for both validation and training data,
    when training the neural network. The validation mode is to be used when
    evaluating the neural network offline, with potentially other
    reconstruction steps used afterwards.
    af (int): the acceleration factor.
    val_split (float): the validation split, between 0 and 1. The split will be
        made on the sessions rather than the files themselves to avoid having
        very similar looking images in the training and the validation sets.
        Defaults to 0.1
    filenames (list): list of the path to the files containing the data you
        want for this particular sequence. When `None`, the files will be looked
        for in the `path`
    seed (int): the random seed used for the validation random split, defaults
        to None

    Raises:
    ValueError: when no nii.gz files can be found in the path directory.
    """
    def __init__(self, path, mode='training', af=4, val_split=0.1, filenames=None, seed=None):
        self.path = path
        self.mode = mode
        self.af = af

        if filenames is None:
            self.filenames = glob.glob(path + '**/*.nii.gz', recursive=True)
            if not self.filenames:
                raise ValueError('No compressed nifti files at path {}'.format(path))
            if val_split > 0:
                sessions = [_get_session_from_filename(filename) for filename in self.filenames]
                n_val = int(len(sessions) * val_split)
                random.seed(seed)
                random.shuffle(sessions)
                val_sessions = sessions[:n_val]
                val_filenames = [filename for filename in self.filenames if _get_session_from_filename(filename) in val_sessions]
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
        """Get the volume from the file at `idx` in `self.filenames`.

        Parameters:
        idx (str): index of the nii.gz file containing the data in `self.filenames`

        Returns:
        ndarray: the volume in NHWC format
        """
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
        idx (str): index of the nii.gz file containing the training data
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
        kspaces = np.empty_like(images, dtype=np.complex64)
        mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
        fourier_op = FFT2(fourier_mask)
        for i, image in enumerate(images):
            kspaces[i] = fourier_op.op(image[..., 0])[..., None]
        mask_batch = np.repeat(fourier_mask[None, ...], len(images), axis=0)
        scale_factor = self.scale_factor
        kspaces_scaled = kspaces * scale_factor
        images_scaled = images * scale_factor
        images_scaled = images_scaled.astype(np.float32)
        return ([kspaces_scaled, mask_batch], images_scaled)

class ZeroFilled2DSequence(Masked2DSequence):
    """
    This sequence generates pre-reconstructed examples, with zero filling.
    """

    def __getitem__(self, idx):
        """Get the reconstructed images and the images of the volume.

        This method will generate a mask on-the-fly, mask the kspaces and then
        do a zero-filled reconstruction.

        Parameters:
        idx (int): index of the nii.gz file containing the training data
            in `self.filenames`.

        Returns:
        tuple (ndarray, ndarray): the reconstructed masked kspaces and the
            images corresponding to the volume in NHWC format.
        """
        [kspaces_scaled, _], images_scaled = super(ZeroFilled2DSequence, self).__getitem__(idx)
        im_z_reco = zero_filled_recon(np.squeeze(kspaces_scaled))[..., None]
        return im_z_reco, images_scaled
