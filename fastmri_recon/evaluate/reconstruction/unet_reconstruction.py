import numpy as np


def reco_unet_from_test_file(zero_img_batch, means, stddevs, model):
    im_recos = model.predict_on_batch(zero_img_batch)
    im_recos = np.squeeze(im_recos)
    im_recos = im_recos * np.array(stddevs)[:, None, None]
    im_recos = im_recos + np.array(means)[:, None, None]
    return im_recos

def reco_and_gt_unet_from_val_file(zero_img_batch, img_batch, means, stddevs, model):
    im_recos = reco_unet_from_test_file(zero_img_batch, means, stddevs, model)
    img_batch = np.squeeze(img_batch)
    img_batch = img_batch * np.array(stddevs)[:, None, None]
    img_batch = img_batch + np.array(means)[:, None, None]
    return im_recos, img_batch

def reco_and_gt_unet_from_val_file_no_norm(zero_img_batch, img_batch, model):
    im_recos = model.predict_on_batch(zero_img_batch)
    im_recos = np.squeeze(im_recos)
    return im_recos, np.squeeze(img_batch)
