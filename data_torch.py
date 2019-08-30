import numpy as np
import torch
from torch.utils.data import Dataset

from data import MaskedUntouched2DSequence, MaskedUntouched2DAllLoadedSequence

class MaskedUntouched2DDataset(MaskedUntouched2DSequence, Dataset):

    def get_item_train(self, filename):
        ([kspaces, mask_batch], images) = super(MaskedUntouched2DDataset, self).get_item_train(filename)
        images = torch.from_numpy(images[..., 0])
        mask_batch = torch.from_numpy(mask_batch)
        kspaces = torch.from_numpy(np.concatenate([kspaces.real, kspaces.imag], axis=-1))
        return kspaces, mask_batch, images

    def get_item_test(self, filename):
        kspaces, mask_batch = super(MaskedUntouched2DDataset, self).get_item_test(filename)
        kspaces = torch.from_numpy(np.concatenate([kspaces.real, kspaces.imag], axis=-1))
        mask_batch = torch.from_numpy(mask_batch)
        return kspaces, mask_batch


class MaskedUntouched2DAllLoadedDataset(MaskedUntouched2DAllLoadedSequence, Dataset):

    def get_item_train(self, filename):
        ([kspaces, mask_batch], images) = super(MaskedUntouched2DAllLoadedDataset, self).get_item_train(filename)
        images = torch.from_numpy(images[..., 0])
        mask_batch = torch.from_numpy(mask_batch)
        kspaces = torch.from_numpy(np.concatenate([kspaces.real, kspaces.imag], axis=-1))
        return kspaces, mask_batch, images

    def get_item_test(self, filename):
        kspaces, mask_batch = super(MaskedUntouched2DAllLoadedDataset, self).get_item_test(filename)
        kspaces = torch.from_numpy(np.concatenate([kspaces.real, kspaces.imag], axis=-1))
        mask_batch = torch.from_numpy(mask_batch)
        return kspaces, mask_batch
