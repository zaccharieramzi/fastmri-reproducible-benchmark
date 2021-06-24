import matplotlib.pyplot as plt
import numpy as np

from fastmri_recon.config import LOGS_DIR
from fastmri_recon.evaluate.metrics.np_metrics import psnr, ssim


plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['image.cmap'] = 'gray'

def save_figure(im_recos, img_batch, name, slice_index=15, acq_type='radial', af=4, three_d=False):
    if three_d:
        im_reco = im_recos[0, slice_index]
        im_gt = img_batch[0, slice_index]
    else:
        im_reco = im_recos[0]
        im_gt = img_batch[0]
    im_gt = im_gt.squeeze()
    im_reco = im_reco.squeeze()
    im_res = np.abs(im_gt - im_reco)
    im_res = np.abs(im_res)
    im_gt = np.abs(im_gt)
    im_reco = np.abs(im_reco)
    p = psnr(im_gt, im_reco)
    s = ssim(im_gt[None], im_reco[None])
    fig, ax = plt.subplots(1, frameon=False)
    ax.imshow(im_reco, aspect='auto')
    ax.text(
        1.0, 1.0,
        f'PSNR: {p:.2f}/ SSIM: {s:.4f}',
        ha='left', va='top',
        fontsize='medium',
        color='white',
        transform=fig.transFigure,
    )
    ax.axis('off')
    fig.savefig(f'{LOGS_DIR}figures/{name}{acq_type}_recon_af{af}.png')
    fig, ax = plt.subplots(1, frameon=False)
    ax.imshow(im_res, aspect='auto')
    ax.axis('off')
    fig.savefig(f'{LOGS_DIR}figures/{name}{acq_type}_residu_af{af}.png')
    fig, ax = plt.subplots(1)
    ax.imshow(im_gt)
    ax.axis('off')
    fig.savefig(f'{LOGS_DIR}figures/image_gt.png')
