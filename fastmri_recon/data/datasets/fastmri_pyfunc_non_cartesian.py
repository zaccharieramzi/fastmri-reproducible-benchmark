import tensorflow as tf
from tfkbnufft.kbnufft import KbNufftModule

from .preprocessing import non_cartesian_from_kspace_to_nc_kspace_and_traj
from ..utils.h5 import from_train_file_to_image_and_kspace_and_contrast


def train_nc_kspace_dataset_from_indexable(
        path,
        image_size,
        inner_slices=None,
        rand=False,
        scale_factor=1,
        contrast=None,
        n_samples=None,
        acq_type='radial',
        compute_dcomp=False,
        **acq_kwargs,
    ):
    r"""Non-cartesian dataset for the training/validation set of single coil fastMRI.

    The undersampling is performed retrospectively on the fully-sampled kspace,
    using spiral or radial trajectories. A uniform cartesian trajectory is also
    available for debugging.
    The non-uniform Fourier transform is implemented by tfkbnufft.
    The output of the dataset is of the form:
    ```
    (retrospectively_undersampled_nc_kspace, nc_trajectory, extra_args), ground_truth_reconstruction
    ```
    where `extra_args` contains the original shape fo the image, and potentially
    the density compensation factors.

    Prefetching is performed, as well as parallel calls for preprocessing when
    rand or inner_slices are active.
    The ground truth reconstruction is read directly from the h5 file and not
    obtained through the Fourier inversion of the kspace.

    Arguments:
        path (str): the path to the fastMRI files. Should end with a `/`.
        image_size (tuple of int): the fixed image size to consider for the
            non-uniform Fourier transform. It needs to be fixed to only use
            a single plan. An image size of (640, 400) will lead to 99.5
            percents of the dataset being only padded and not cropped.
        inner_slices (int or None): the slices to consider in the volumes. The
            slicing will be performed as `inner_slices//2:-inner_slices//2`.
            If None, all slices are considered. Defaults to None.
        rand (bool): whether or not to sample one slice randomly from the
            considered slices of the volumes. If None, all slices are taken.
            Defaults to None.
        scale_factor (int or float): the multiplicative scale factor for both the
            kspace and the target reconstruction. Typically, 1e6 is a good value
            for fastMRI, because it sets the values in a range acceptable for
            neural networks training. See [R2020] (3.4 Training) for more details
            on this value. Defaults to 1.
        contrast (str or None): the contrast to select for this dataset. If None,
            all contrasts are considered. Available contrasts for fastMRI single
            coil are typically `CORPD_FBK` (Proton density) and `CORPDFS_FBK`
            (Proton density with fat suppression). Defaults to None.
        n_samples (int or None): the number of samples to consider from this set.
            If None, all samples are used. Defaults to None.
        acq_type (str): the type of non-cartesian trajectory to use. Choices are
            `radial`, `spiral` or `cartesian` (for debugging purposes). Defaults
            to radial.
        compute_dcomp (bool): whether to compute and return the density compensation
            factors. See [P1999] for more on density compensation. Defaults to
            False.
        **acq_kwargs: keyword arguments for the non-cartesian trajectory. See
            fastmri_recon/data/utils/non_cartesian.py for more info.

    Returns:
        tf.data.Dataset: the training/validation non-cartesian dataset.
    """
    selection = [{'inner_slices': inner_slices, 'rand': rand}]
    def _tf_filename_to_image_and_kspace_and_contrast(filename):
        def _from_train_file_to_image_and_kspace_and_contrast_tensor_to_tensor(filename):
            filename_str = filename.numpy()
            image, kspace, contrast = from_train_file_to_image_and_kspace_and_contrast(
                filename_str,
                selection=selection,
            )
            return tf.convert_to_tensor(image), tf.convert_to_tensor(kspace), tf.convert_to_tensor(contrast)
        [image, kspace, contrast] = tf.py_function(
            _from_train_file_to_image_and_kspace_and_contrast_tensor_to_tensor,
            [filename],
            [tf.float32, tf.complex64, tf.string],
        )
        if rand:
            n_slices = (1,)
        else:
            n_slices = (inner_slices,)
        kspace_size = n_slices + (640, None)
        image_size = n_slices + (320, 320)
        image.set_shape(image_size)
        kspace.set_shape(kspace_size)
        return image, kspace, contrast

    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', shuffle=False)
    # this makes sure the file selection is happening once when using less than
    # all samples
    files_ds = files_ds.shuffle(
        buffer_size=1000,
        seed=0,
        reshuffle_each_iteration=False,
    )
    image_and_kspace_and_contrast_ds = files_ds.map(
        _tf_filename_to_image_and_kspace_and_contrast,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # contrast filtering
    if contrast:
        image_and_kspace_and_contrast_ds = image_and_kspace_and_contrast_ds.filter(
            lambda image, kspace, tf_contrast: tf_contrast == contrast
        )
    image_and_kspace_ds = image_and_kspace_and_contrast_ds.map(
        lambda image, kspace, tf_contrast: (image, kspace),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if n_samples is not None:
        image_and_kspace_ds = image_and_kspace_ds.take(n_samples)
    nufft_ob = KbNufftModule(
        im_size=image_size,
        grid_size=None,
        norm='ortho',
    )
    masked_kspace_ds = image_and_kspace_ds.map(
        non_cartesian_from_kspace_to_nc_kspace_and_traj(
            nufft_ob,
            image_size,
            acq_type=acq_type,
            scale_factor=scale_factor,
            compute_dcomp=compute_dcomp,
            **acq_kwargs,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if rand or inner_slices is not None else None,
    ).repeat()
    if rand or inner_slices is not None:
        masked_kspace_ds = masked_kspace_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds
