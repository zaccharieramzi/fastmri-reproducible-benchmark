import numpy as np


def reco_net_from_test_file(kspace_and_mask_batch, model):
    im_recos = model.predict_on_batch(kspace_and_mask_batch)
    im_recos = np.squeeze(im_recos)
    return im_recos

def reco_and_gt_net_from_val_file(kspace_and_mask_batch, img_batch, model):
    im_recos = reco_net_from_test_file(kspace_and_mask_batch, model)
    img_batch = np.squeeze(img_batch)
    return im_recos, img_batch
