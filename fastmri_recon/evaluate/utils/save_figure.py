import matplotlib.pyplot as plt
import numpy as np

from fastmri_recon.config import LOGS_DIR


plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['image.cmap'] = 'gray'

def save_figure(im_recos, img_batch, name, slice_index=15, acq_type='radial', af=4, three_d=False):
    if three_d:
        im_reco = im_recos[0, slice_index]
        im_gt = img_batch[0, slice_index]
    else:
        im_reco = im_recos[0]
        im_gt = img_batch[0]
    im_res = np.abs(im_gt - im_reco)
    fig, ax = plt.subplots(1, frameon=False)
    ax.imshow(np.abs(np.squeeze(im_reco)), aspect='auto')
    ax.axis('off')
    fig.savefig(f'{LOGS_DIR}figures/{name}{acq_type}_recon_af{af}.png')
    fig, ax = plt.subplots(1, frameon=False)
    ax.imshow(np.abs(np.squeeze(im_res)), aspect='auto')
    ax.axis('off')
    fig.savefig(f'{LOGS_DIR}figures/{name}{acq_type}_residu_af{af}.png')
    fig, ax = plt.subplots(1)
    ax.imshow(np.abs(np.squeeze(im_gt)))
    ax.axis('off')
    fig.savefig(f'{LOGS_DIR}figures/image_gt.png')
