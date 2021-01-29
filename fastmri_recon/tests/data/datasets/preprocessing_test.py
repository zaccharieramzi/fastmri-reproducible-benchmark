import tensorflow as tf
from tfkbnufft.kbnufft import KbNufftModule

from fastmri_recon.data.datasets.preprocessing import non_cartesian_from_kspace_to_nc_kspace_and_traj

def test_gridded_preprocessing():
    image_size = (64, 40)
    nfft_ob = KbNufftModule(im_size=image_size)
    preproc_fun = non_cartesian_from_kspace_to_nc_kspace_and_traj(
        nfft_ob,
        image_size,
        gridding=True,
        af=8,
    )
    image = tf.random.normal([1, 32, 32])
    kspace = tf.cast(tf.random.normal([1, 64, 32]), tf.complex64)
    (kspace_masked, mask), image_out = preproc_fun(image, kspace)
