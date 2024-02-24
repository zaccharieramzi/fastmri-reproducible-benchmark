import glob
import os.path as op
import random
import re

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import Sequence

from ...evaluate.reconstruction.zero_filled_reconstruction import zero_filled_recon
from ..utils.masking.gen_mask import gen_mask
from ..utils.fourier import FFT2


def _get_subject_from_filename(filename):
    base_name = op.basename(filename)
    subject_id = re.findall(r'OAS3\d{4}', base_name)[0]
    return subject_id

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
        made on the subjects rather than the files themselves to avoid having
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
    def __init__(self, path, mode='training', af=4, val_split=0.1, filenames=None, seed=None, reorder=True):
        self.path = path
        self.mode = mode
        self.af = af
        self.reorder = reorder

        if filenames is None:
            self.filenames = glob.glob(path + '**/*.nii.gz', recursive=True)
            if not self.filenames:
                raise ValueError('No compressed nifti files at path {}'.format(path))
            if val_split > 0:
                subjects = [_get_subject_from_filename(filename) for filename in self.filenames]
                n_val = int(len(subjects) * val_split)
                random.seed(seed)
                random.shuffle(subjects)
                val_subjects = subjects[:n_val]
                val_filenames = [filename for filename in self.filenames if _get_subject_from_filename(filename) in val_subjects]
                self.filenames = [filename for filename in self.filenames if filename not in val_filenames]
                self.val_sequence = type(self)(path, mode=mode, af=af, val_split=0, filenames=val_filenames, reorder=reorder)
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
        # this is necessary because the data is not necessarily ordered with slices first
        if self.reorder and min(images.shape) != images.shape[0]:
            images = np.moveaxis(images, -1, 0)
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
            slice_start = max(n_slices // 2 - self.inner_slices // 2, 0)
            slice_end = min(slice_start + self.inner_slices, n_slices)
            if self.rand:
                i_slice = random.randint(slice_start, slice_end - 1)
                selected_slices = slice(i_slice, i_slice + 1)
            else:
                selected_slices = slice(slice_start, slice_end)
        images = images[selected_slices]
        k_shape = images[0].shape
        kspaces = np.empty_like(images, dtype=np.complex64)
        mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float32), k_shape[0], axis=0)
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

    def __init__(self, *args, n_pooling=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_pooling = n_pooling
        if self.val_sequence is not None:
            self.val_sequence.n_pooling = n_pooling

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
        im_z_reco = zero_filled_recon(kspaces_scaled[..., 0])[..., None]
        if self.n_pooling > 1:
            im_shape = images_scaled.shape[1:3]
            if any(image_dim % 2**self.n_pooling != 0 for image_dim in im_shape):
                im_z_reco = self._pad_image(im_z_reco)
                images_scaled = self._pad_image(images_scaled)
        return im_z_reco, images_scaled

    def _pad_image(self, img):
        pool = self.n_pooling
        im_shape = np.array(img.shape[1:3])
        to_pad = ((im_shape / 2**pool).astype(int) + 1) * 2**pool - im_shape
        pad_seq = [(0, 0), (to_pad[0]//2, to_pad[0]//2), (to_pad[1]//2, to_pad[1]//2), (0, 0)]
        img_padded = np.pad(img, pad_seq, mode='constant')
        return img_padded


class KIKISequence(Oasis2DSequence):
    """This sequence allows to generate a mask on-the-fly when enumerating
    training or validation examples. It also allows you to restrict the
    training to only innermost parts of the volumes, and select randomly
    a slice when training. Finally, you can scale the values of the
    kspaces and images by a factor.
    The target values are not cropped or in magnitude, but the actual ones.

    Parameters:
    inner_slices (int): the number of inner slices you want to consider when
    enumerating the volumes.
    rand (bool): whether you want to only pick one random slice from the
    considered slices when enumerating the volumes.
    scale_factor (float): the factor by which to multiply the kspaces and the
    images, if scaling is needed
    space (str): the space of the sequence, i.e. whether the target value is
    the ground truth k-space (K) or the ground-truth image (I).
    """
    def __init__(self, *args, inner_slices=None, rand=False, scale_factor=1, space='K', **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_slices = inner_slices
        self.rand = rand
        self.scale_factor = scale_factor
        self.space = space
        if self.val_sequence is not None:
            self.val_sequence.inner_slices = inner_slices
            self.val_sequence.scale_factor = scale_factor

    def __getitem__(self, idx):
        """Get a training triplet from the file at filename.

        This method will get the kspaces and images at filename, create a mask
        on-the-fly, mask the kspaces with it, select only the relevant slices,
        and return a tuple ((kspaces, mask), images).

        Parameters:
        filename (str): the name of the h5 file containing the images and
        the kspaces.

        Returns:
        tuple ((ndarray, ndarray), ndarray): the masked kspaces, mask and images
        corresponding to the volume in NHWC format (mask is NHW).
        """
        images = super(KIKISequence, self).__getitem__(idx)
        if self.inner_slices is not None:
            n_slices = len(images)
            slice_start = max(n_slices // 2 - self.inner_slices // 2, 0)
            slice_end = min(slice_start + self.inner_slices, n_slices)
            if self.rand:
                i_slice = random.randint(slice_start, slice_end - 1)
                selected_slices = slice(i_slice, i_slice + 1)
            else:
                selected_slices = slice(slice_start, slice_end)
        images = images[selected_slices]
        k_shape = images[0].shape
        kspaces = np.empty_like(images, dtype=np.complex64)
        kspaces_masked = np.empty_like(images, dtype=np.complex64)
        mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
        fourier_mask = np.repeat(mask.astype(np.float32), k_shape[0], axis=0)
        fourier_op = FFT2(np.array([1]))
        for i, image in enumerate(images):
            kspaces[i] = fourier_op.op(image[..., 0])[..., None]
            kspaces_masked[i] = kspaces[i] * fourier_mask[..., None]
        mask_batch = np.repeat(fourier_mask[None, ...], len(images), axis=0)
        scale_factor = self.scale_factor
        kspaces_scaled = kspaces * scale_factor
        kspaces_masked_scaled = kspaces_masked * scale_factor
        images_scaled = images * scale_factor
        images_scaled = images_scaled.astype(np.float32)
        if self.space == 'K':
            return ([kspaces_masked_scaled, mask_batch], kspaces_scaled)
        elif self.space == 'I':
            images = zero_filled_recon(kspaces_scaled[..., 0])[..., None]
            return ([kspaces_masked_scaled, mask_batch], images)
