import tensorflow as tf

from .preprocessing import generic_from_kspace_to_masked_kspace_and_mask, generic_prepare_mask_and_kspace
from ..utils.h5 import from_train_file_to_image_and_kspace_and_contrast, from_test_file_to_mask_and_kspace_and_contrast
from ..utils.masking.acceleration_factor import tf_af

# TODO: add unet and kikinet datasets
def tf_filename_to_mask_and_kspace_and_contrast(filename):
    def _from_test_file_to_mask_and_kspace_and_contrast_tensor_to_tensor(filename):
        filename_str = filename.numpy()
        mask, kspace, contrast = from_test_file_to_mask_and_kspace_and_contrast(filename_str)
        return tf.convert_to_tensor(mask), tf.convert_to_tensor(kspace), tf.convert_to_tensor(contrast)
    [mask, kspace, contrast] = tf.py_function(
        _from_test_file_to_mask_and_kspace_and_contrast_tensor_to_tensor,
        [filename],
        [tf.bool, tf.complex64, tf.string],
    )
    mask.set_shape((None,))
    kspace.set_shape((None, 640, None))
    return mask, kspace, contrast


def train_masked_kspace_dataset_from_indexable(
        path,
        AF=4,
        inner_slices=None,
        rand=False,
        scale_factor=1,
        contrast=None,
        n_samples=None,
        fixed_masks=False,
        batch_size=None,
        target_image_size=(640, 400),
        mask_type='random',
        input_context=None,
    ):
    r"""Dataset for the training/validation set of single coil fastMRI.

    The undersampling is performed retrospectively on the fully-sampled kspace,
    using the cartesian masks described in [Z2018] for the knee dataset. These
    masks have an autocalibration region whose width depends on the acceleration
    factor and sample randomly in the high frequencies.
    The output of the dataset is of the form:
    ```
    (retrospectively_undersampled_kspace, undersampling_mask), ground_truth_reconstruction
    ```

    Prefetching is performed, as well as parallel calls for preprocessing.
    The ground truth reconstruction is read directly from the h5 file and not
    obtained through the Fourier inversion of the kspace.

    Arguments:
        path (str): the path to the fastMRI files. Should end with a `/`.
        AF (int): the acceleration factor, generally 4 or 8 for fastMRI.
            Defaults to 4.
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
        fixed_masks (bool): whether or not to use a single mask for all the
            training. A caveat is that there are different masks for different
            shapes, but for a given shape, only one mask is used.

    Returns:
        tf.data.Dataset: the training/validation dataset.
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
    if input_context is not None:
        files_ds = files_ds.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    # this makes sure the file selection is happening once when using less than
    # all samples
    files_ds = files_ds.shuffle(
        buffer_size=1000 if input_context is None else 1000 // input_context.num_input_pipelines,
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
    masked_kspace_ds = image_and_kspace_ds.map(
        generic_from_kspace_to_masked_kspace_and_mask(
            AF=AF,
            scale_factor=scale_factor,
            fixed_masks=fixed_masks,
            batch_size=batch_size,
            target_image_size=target_image_size,
            mask_type=mask_type,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if batch_size is not None:
        masked_kspace_ds = masked_kspace_ds.batch(batch_size)
    masked_kspace_ds = masked_kspace_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds

def test_masked_kspace_dataset_from_indexable(path, AF=4, scale_factor=1, contrast=None, n_samples=None):
    r"""Dataset for the testing/challenge set of single coil fastMRI.

    The output of the dataset is of the form:
    ```
    undersampled_kspace, undersampling_mask
    ```

    Prefetching is performed, as well as parallel calls for preprocessing.

    Arguments:
        path (str): the path to the fastMRI files. Should end with a `/`.
        AF (int): the acceleration factor, 4 or 8. A kspace is deemed to have
            an acceleration factor of 4 if its actual acceleration factor is below
            5.5. It will be deemed to have an acceleration afctor of 8 otherwise.
            Defaults to 4.
        scale_factor (int or float): the multiplicative scale factor for both the
            kspace and the target reconstruction. Typically, 1e6 is a good value
            for fastMRI, because it sets the values in a range acceptable for
            neural networks training. See [R2020] (3.4 Training) for more details
            on this value). Don't forget to use the same scaling factor for
            training, evaluation and inference. Defaults to 1.
        contrast (str or None): the contrast to select for this dataset. If None,
            all contrasts are considered. Available contrasts for fastMRI single
            coil are typically `CORPD_FBK` (Proton density) and `CORPDFS_FBK`
            (Proton density with fat suppression). Defaults to None.

    Returns:
        tf.data.Dataset: the testing/challenge dataset.
    """
    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', seed=0, shuffle=False)
    mask_and_kspace_and_contrast_ds = files_ds.map(
        tf_filename_to_mask_and_kspace_and_contrast,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # contrast filtering
    if contrast:
        mask_and_kspace_and_contrast_ds = mask_and_kspace_and_contrast_ds.filter(
            lambda mask, kspace, tf_contrast: tf_contrast == contrast
        )
    mask_and_kspace_ds = mask_and_kspace_and_contrast_ds.map(
        lambda mask, kspace, tf_contrast: (mask, kspace),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # af filtering
    if AF == 4:
        mask_and_kspace_ds = mask_and_kspace_ds.filter(
            lambda mask, kspace: tf_af(mask) < 5.5
        )
    else:
        mask_and_kspace_ds = mask_and_kspace_ds.filter(
            lambda mask, kspace: tf_af(mask) > 5.5
        )
    if n_samples is not None:
        mask_and_kspace_ds = mask_and_kspace_ds.take(n_samples)
    masked_kspace_ds = mask_and_kspace_ds.map(
        generic_prepare_mask_and_kspace(
            scale_factor=scale_factor,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return masked_kspace_ds
