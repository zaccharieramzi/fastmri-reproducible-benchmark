import numpy as np


def gen_mask(kspace, accel_factor=8, seed=None):
    # inspired by https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
    shape = kspace.shape
    num_cols = shape[-1]

    center_fraction = (32 // accel_factor) / 100
    acceleration = accel_factor

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = np.random.default_rng(seed).uniform(size=num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)
    return mask

def gen_mask_equidistant(kspace, accel_factor=8, seed=None):
    # inspired by https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
    shape = kspace.shape
    num_cols = shape[-1]

    center_fraction = (32 // accel_factor) / 100
    acceleration = accel_factor

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    num_high_freqs = num_cols // accel_factor - num_low_freqs
    high_freqs_spacing = (num_cols - num_low_freqs) // num_high_freqs
    acs_lim = (num_cols - num_low_freqs + 1) // 2
    mask_offset = random.randrange(high_freqs_spacing)
    high_freqs_location = np.arange(mask_offset, num_cols, high_freqs_spacing)
    low_freqs_location = np.arange(acs_lim, acs_lim + num_low_freqs)
    mask_locations = np.concatentae(high_freqs_location, low_freqs_location)
    mask = np.zeros((num_cols,))
    mask[mask_locations] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)
    return mask

def gen_mask_vd(kspace, accel_factor=8):
    shape = kspace.shape
    num_cols = shape[-1]

    center_fraction = (32 // accel_factor) / 100
    acceleration = accel_factor

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - 3 * num_low_freqs)
    num_sampling = int(prob * num_cols)
    selected_indexes = np.random.default_rng().randint(0, num_cols, size=(num_sampling, 2)).mean(axis=-1).astype('int')
    mask = np.zeros((num_cols,)).astype('bool')
    mask[selected_indexes] = True
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)
    return mask
