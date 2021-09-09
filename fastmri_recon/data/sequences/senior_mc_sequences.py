import glob
import os.path as op
import random
import re

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import Sequence

from ..utils.masking.gen_mask import gen_mask
from ..utils.fourier import FFT2

""" For Senior data """
def _get_subject_from_filename(filename):
    subject_id = op.basename(op.dirname(filename+'/*'))
    return subject_id

def from_file_to_volume(filename, reverse=True):
    volume = nib.load(filename)
    volume = volume.get_fdata()
    volume = np.rot90(volume) #to keep in conventional axis mode 
    if reverse and min(volume.shape) != volume.shape[0]:
        volume = np.moveaxis(volume, -1, 0)
    volume = volume[..., None]    
    volume = volume.astype(np.complex64)
    return volume

class Senior_MC_2DSequence(Sequence):
  
    
    def __init__(self, path, mode='training', af=4, val_split=0.1, filenames=None, seed=None, reorder=True):
        self.path = path
        self.mode = mode
        self.af = af
        self.reorder = reorder

        if filenames is None:
            self.filenames = glob.glob(path+'/*/*')         

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
        ndarray: the volume in Mc_H_W_Slice format
        """
        filename = glob.glob(self.filenames[idx]+'/*')
        images = []
        for i in range(len(filename)):
            image = from_file_to_volume(filename[i])
            images.append(image)
        
        return np.asarray(images)


class Masked_MC_2DSequence(Senior_MC_2DSequence):
    def __init__(self, *args, inner_slices=None, rand=False, scale_factor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_slices = inner_slices
        self.rand = rand
        self.scale_factor = scale_factor
        if self.val_sequence is not None:
            self.val_sequence.inner_slices = inner_slices
            self.val_sequence.scale_factor = scale_factor

    def __getitem__(self, idx):

        images = super(Masked_MC_2DSequence, self).__getitem__(idx)

        
        if self.inner_slices is not None:
            n_slices = len(images[0])
            slice_start = max(n_slices // 2 - self.inner_slices // 2, 0)
            slice_end = min(slice_start + self.inner_slices, n_slices)
            if self.rand:
                i_slice = random.randint(slice_start, slice_end - 1)
                selected_slices = slice(i_slice, i_slice + 1)
            else:
                selected_slices = slice(slice_start, slice_end)
              
        mask_batches = []
        kspaces_img = []
        img = []         
        for i in range(len(images)):
            image = images[i]
            image = image[selected_slices]
            k_shape = image[0].shape
            kspaces = np.empty_like(image, dtype=np.complex64)
            mask = gen_mask(kspaces[0, ..., 0], accel_factor=self.af)
            fourier_mask = np.repeat(mask.astype(np.float), k_shape[0], axis=0)
            fourier_op = FFT2(fourier_mask)
            for i, img1 in enumerate(image):
                kspaces[i] = fourier_op.op(img1[..., 0])[..., None]
                
            mask_batch = np.repeat(fourier_mask[None, ...], len(image), axis=0)    
            mask_batches.append(mask_batch)
            kspaces_img.append(kspaces)
            img.append(image)
        
        img = np.asarray(img)
        img = np.moveaxis(img, 0, 1)
        
        mask_batches = np.asarray(mask_batches)
        mask_batches = np.moveaxis(mask_batches, 0, 1)
        
        kspaces_img = np.asarray(kspaces_img)
        kspaces_img = np.moveaxis(kspaces_img, 0, 1)
        
        scale_factor = self.scale_factor
        kspaces_scaled = kspaces_img * scale_factor
        images_scaled = img * scale_factor
        images_scaled = images_scaled.astype(np.float32)
        return ([kspaces_scaled, mask_batches], images_scaled)


