"""torch Datasets used for fastMRI data"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .fastmri_sequences import Masked2DSequence

class Masked2DDataset(Masked2DSequence, Dataset):
    """Sequence equivalent for torch. Read Masked2DSequence docs for more.
    This dataset concatenates real and imaginary parts on a 5th axis because
    complex numbers are not handled in pytorch.
    """
    def get_item_train(self, filename):
        ([kspaces, mask_batch], images) = super(Masked2DDataset, self).get_item_train(filename)
        images = torch.from_numpy(images[..., 0])
        mask_batch = torch.from_numpy(mask_batch)
        kspaces = torch.from_numpy(np.concatenate([kspaces.real, kspaces.imag], axis=-1))
        return kspaces, mask_batch, images

    def get_item_test(self, filename):
        kspaces, mask_batch = super(Masked2DDataset, self).get_item_test(filename)
        kspaces = torch.from_numpy(np.concatenate([kspaces.real, kspaces.imag], axis=-1))
        mask_batch = torch.from_numpy(mask_batch)
        return kspaces, mask_batch
