import tensorflow as tf
from tfkbnufft.kbnufft import KbNufftModule

from fastmri_recon.data.datasets.preprocessing import non_cartesian_from_kspace_to_nc_kspace_and_traj


tf.config.experimental_run_functions_eagerly(True)

def test_gridded_preprocessing():
    image_size = (640, 400)
    nfft_ob = KbNufftModule(im_size=image_size)
    preproc_fun = non_cartesian_from_kspace_to_nc_kspace_and_traj(
        nfft_ob,
        image_size,
        gridding=True,
        af=20,
    )
    image = tf.random.normal([1, 320, 320])
    kspace = tf.cast(tf.random.normal([1, 640, 320]), tf.complex64)
    (kspace_masked, mask), image_out = preproc_fun(image, kspace)
    assert tf.squeeze(kspace_masked).shape == image_size
