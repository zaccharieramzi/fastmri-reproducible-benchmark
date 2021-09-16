import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from fastmri_recon.config import LOGS_DIR
from fastmri_recon.evaluate.metrics.np_metrics import psnr, ssim


plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['image.cmap'] = 'gray'

def save_figure(
        im_recos,
        img_batch,
        name,
        slice_index=15,
        acq_type='radial',
        af=4,
        three_d=False,
        zoom=None,
        draw_zoom=None,
        brain=False,
    ):
    if three_d:
        p = psnr(img_batch[0].numpy().squeeze(), im_recos[0].squeeze())
        im_reco = im_recos[0, slice_index]
        im_gt = img_batch[0, slice_index]
    else:
        p = psnr(img_batch.numpy().squeeze(), im_recos.squeeze())
        s = ssim(img_batch.numpy().squeeze(), im_recos.squeeze())
        im_reco = im_recos[slice_index]
        im_gt = img_batch[slice_index]
    im_gt = im_gt.numpy().squeeze()
    im_reco = im_reco.squeeze()
    if brain:
        im_gt = im_gt[::-1]
        im_reco = im_reco[::-1]
    if three_d:
        # we rotate the image 90 degrees counter-clockwise
        im_gt = np.rot90(im_gt)
        im_reco = np.rot90(im_reco)
    if zoom is not None:
        name += '_zoom'
        im_gt = im_gt[zoom[1][0]:zoom[1][1], zoom[0][0]:zoom[0][1]]
        im_reco = im_reco[zoom[1][0]:zoom[1][1], zoom[0][0]:zoom[0][1]]
    im_res = np.abs(im_gt - im_reco)
    im_res = np.abs(im_res)
    im_gt = np.abs(im_gt)
    im_reco = np.abs(im_reco)

    fig, ax = plt.subplots(1, frameon=False)
    ax.imshow(im_reco, aspect='auto')
    if zoom is None:
        text = f'PSNR: {p:.2f}'
        if not three_d:
            text += f'/ SSIM: {s:.4f}'
        ax.text(
            1.0, 1.0,
            text,
            ha='left', va='top',
            fontsize='x-large',
            color='red',
        )
    if draw_zoom is not None:
        rect = patches.Rectangle(
            (draw_zoom[0][0], draw_zoom[1][0]),
            draw_zoom[0][1] - draw_zoom[0][0], draw_zoom[1][1] - draw_zoom[1][0],
            linewidth=1,
            edgecolor='r',
            facecolor='none',
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.axis('off')
    fig.savefig(f'{LOGS_DIR}figures/{name}{acq_type}_recon_af{af}.png')
    fig, ax = plt.subplots(1, frameon=False)
    ax.imshow(im_res, aspect='auto')
    ax.axis('off')
    fig.savefig(f'{LOGS_DIR}figures/{name}{acq_type}_residu_af{af}.png')
    fig, ax = plt.subplots(1)
    ax.imshow(im_gt)
    if draw_zoom is not None:
        rect = patches.Rectangle(
            (draw_zoom[0][0], draw_zoom[1][0]),
            draw_zoom[0][1] - draw_zoom[0][0], draw_zoom[1][1] - draw_zoom[1][0],
            linewidth=1,
            edgecolor='r',
            facecolor='none',
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.axis('off')
    if zoom is None:
        fig.savefig(f'{LOGS_DIR}figures/image_gt.png')
    else:
        fig.savefig(f'{LOGS_DIR}figures/image_gt_zoom.png')
