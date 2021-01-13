from math import pi

import tensorflow as tf

from fastmri_recon.data.utils.fourier import tf_ortho_ifft2d


@tf.function
def extract_smaps(kspace, low_freq_percentage=8, background_thresh=4e-6):
    """Extract raw sensitivity maps for kspaces

    This function will first select a low frequency region in all the kspaces,
    then Fourier invert it, and finally perform a normalisation by the root
    sum-of-square.
    kspace has to be of shape: nslices x ncoils x height x width

    Arguments:
        kspace (tf.Tensor): the kspace whose sensitivity maps you want extracted.
        low_freq_percentage (int): the low frequency region to consider for
            sensitivity maps extraction, given as a percentage of the width of
            the kspace. In fastMRI, it's 8 for an acceleration factor of 4, and
            4 for an acceleration factor of 8. Defaults to 8.
        background_thresh (float): unused for now, will later allow to have
            thresholded sensitivity maps.

    Returns:
        tf.Tensor: extracted raw sensitivity maps.
    """
    n_slices = tf.shape(kspace)[0]
    if n_slices > 0:
        n_low_freq = tf.cast(tf.shape(kspace)[-2:] * low_freq_percentage / 100, tf.int32)
        center_dimension = tf.cast(tf.shape(kspace)[-2:] / 2, tf.int32)
        low_freq_lower_locations = center_dimension - tf.cast(n_low_freq / 2, tf.int32)
        low_freq_upper_locations = center_dimension + tf.cast(n_low_freq / 2, tf.int32)
        ###
        # NOTE: the following stands for in numpy:
        # low_freq_mask = np.zeros_like(kspace)
        # low_freq_mask[
        #     ...,
        #     low_freq_lower_locations[0]:low_freq_upper_locations[0],
        #     low_freq_lower_locations[1]:low_freq_upper_locations[1]
        # ] = 1
        x_range = tf.range(low_freq_lower_locations[0], low_freq_upper_locations[0])
        y_range = tf.range(low_freq_lower_locations[1], low_freq_upper_locations[1])
        X_range, Y_range = tf.meshgrid(x_range, y_range)
        X_range = tf.reshape(X_range, (-1,))
        Y_range = tf.reshape(Y_range, (-1,))
        low_freq_mask_indices = tf.stack([X_range, Y_range], axis=-1)
        # we have to transpose because only the first dimension can be indexed in
        # scatter_nd
        scatter_nd_perm = [2, 3, 0, 1]
        low_freq_mask = tf.scatter_nd(
            indices=low_freq_mask_indices,
            updates=tf.ones([
                tf.size(X_range),
                tf.shape(kspace)[0],
                tf.shape(kspace)[1]],
            ),
            shape=[tf.shape(kspace)[i] for i in scatter_nd_perm],
        )
        low_freq_mask = tf.transpose(low_freq_mask, perm=scatter_nd_perm)
        ###
        low_freq_kspace = kspace * tf.cast(low_freq_mask, kspace.dtype)
        coil_image_low_freq = tf_ortho_ifft2d(low_freq_kspace)
        # no need to norm this since they all have the same norm
        low_freq_rss = tf.norm(coil_image_low_freq, axis=1)
        coil_smap = coil_image_low_freq / low_freq_rss[:, None]
        # for now we do not perform background removal based on low_freq_rss
        # could be done with 1D k-means or fixed background_thresh, with tf.where
    else:
        coil_smap = tf.zeros_like(kspace, dtype=kspace.dtype)
    return coil_smap


def non_cartesian_extract_smaps(kspace, trajs, dcomp, nufft_back, shape, low_freq_percentage=8):
    def _crop_for_pad(image, shape, im_size):
        to_pad = im_size[-1] - shape[0]
        cropped_image = image[..., to_pad//2:-to_pad//2]
        return cropped_image
    cutoff_freq = low_freq_percentage / 200 * tf.constant(pi)
    # Get the boolean mask for low frequency
    low_freq_bool_mask = tf.math.reduce_all(tf.math.less_equal(tf.abs(trajs[0]), cutoff_freq), axis=0)
    # Obtain the trajectory, kspace and density compensation for low frequency
    low_freq_traj = tf.boolean_mask(trajs, low_freq_bool_mask, axis=2)
    low_freq_kspace = tf.boolean_mask(kspace, low_freq_bool_mask, axis=2)
    low_freq_dcomp = tf.boolean_mask(dcomp, low_freq_bool_mask, axis=1)[:, None]
    coil_smap = nufft_back(low_freq_kspace * tf.cast(low_freq_dcomp, kspace.dtype), low_freq_traj)
    coil_smap = tf.cond(
        tf.math.greater_equal(shape[0], coil_smap.shape[-1]),
        lambda: coil_smap,
        lambda: _crop_for_pad(coil_smap, shape, coil_smap.shape),
    )
    low_freq_rss = tf.norm(coil_smap, axis=1)
    coil_smap = coil_smap / low_freq_rss[:, None]
    return coil_smap
