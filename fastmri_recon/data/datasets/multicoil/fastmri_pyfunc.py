import tensorflow as tf

from .preprocessing import generic_from_kspace_to_masked_kspace_and_mask, generic_prepare_mask_and_kspace
from ...utils.h5 import from_multicoil_train_file_to_image_and_kspace_and_contrast, from_test_file_to_mask_and_kspace_and_contrast_and_image_size, from_test_file_to_mask_and_contrast
from ...utils.masking.acceleration_factor import tf_af

# TODO: add unet and kikinet datasets

def tf_filename_to_mask_and_kspace_and_contrast_and_image_size(filename):
    def _from_test_file_to_mask_and_kspace_and_contrast_and_image_size_tensor_to_tensor(filename):
        filename_str = filename.numpy()
        mask, kspace, contrast, image_size = from_test_file_to_mask_and_kspace_and_contrast_and_image_size(filename_str)
        return tf.convert_to_tensor(mask), tf.convert_to_tensor(kspace), tf.convert_to_tensor(contrast), tf.convert_to_tensor(image_size)
    [mask, kspace, contrast, image_size] = tf.py_function(
        _from_test_file_to_mask_and_kspace_and_contrast_and_image_size_tensor_to_tensor,
        [filename],
        [tf.bool, tf.complex64, tf.string, tf.int32],
    )
    mask.set_shape((None,))
    kspace.set_shape((None, None, None, None))
    image_size.set_shape((2,))
    return mask, kspace, contrast, image_size

def tf_filename_to_mask_and_contrast_and_filename(filename):
    def _from_test_file_to_mask_and_contrast_tensor_to_tensor(filename):
        filename_str = filename.numpy()
        mask, contrast = from_test_file_to_mask_and_contrast(filename_str)
        return tf.convert_to_tensor(mask), tf.convert_to_tensor(contrast)
    [mask, contrast] = tf.py_function(
        _from_test_file_to_mask_and_contrast_tensor_to_tensor,
        [filename],
        [tf.bool, tf.string],
    )
    mask.set_shape((None,))
    return mask, contrast, filename


def train_masked_kspace_dataset_from_indexable(
        path,
        AF=4,
        inner_slices=None,
        rand=False,
        scale_factor=1,
        contrast=None,
        n_samples=None,
        parallel=True,
        fixed_masks=False,
        output_shape_spec=False,
        mask_type='random',
        batch_size=None,
        target_image_size=(640, 400),
        input_context=None,
    ):
    r"""Dataset for the training/validation set of multi-coil fastMRI.

    The undersampling is performed retrospectively on the fully-sampled kspace,
    using the cartesian masks described in [Z2018]. These
    masks have an autocalibration region whose width depends on the acceleration
    factor and sample randomly in the high frequencies when random or periodicly
    when equidistant.
    The output of the dataset is of the form:
    ```
    model_inputs, ground_truth_reconstruction
    ```
    `model_inputs` begins with `retrospectively_undersampled_kspace, undersampling_mask`.
    It then features  `sensitivity_maps` when `parallel` is False. Finally,
    you can find the `specified_output_shape` when `output_shape_spec` is True.
    This is useful in the case of the brain data because the output shape is
    not the same for each volume.

    The sensitivity maps are extracted without any specific logic. They are raw.
    For more information refer to fastmri_recon/data/utils/multicoil/smap_extract.py.

    Prefetching is performed, as well as parallel calls for preprocessing when
    rand or inner_slices are active.
    The ground truth reconstruction is read directly from the h5 file and not
    obtained through the Fourier inversion of the kspace followed by RSS.

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
        parallel (bool): whether to output only one coil (selected randomly) out
            of all the available coils. Defaults to True.
        fixed_masks (bool): whether or not to use a single mask for all the
            training. A caveat is that there are different masks for different
            shapes, but for a given shape, only one mask is used.
        output_shape_spec (bool): whether you need the output shape to be
            present in the model inputs. It is inferred from the ground truth
            reconstruction present in the file. Defaults to False.
        mask_type (str): the type of mask to use to retrospectively undersample
            the data. Can be 'random' or 'equidistant'. Defaults to 'random'.

    Returns:
        tf.data.Dataset: the training/validation multi-coil dataset.
    """
    selection = [
        {'inner_slices': inner_slices, 'rand': rand},  # slice selection
        {'rand': parallel, 'keep_dim': False},  # coil selection
    ]
    def _tf_filename_to_image_and_kspace_and_contrast(filename):
        def _from_train_file_to_image_and_kspace_and_contrast_tensor_to_tensor(filename):
            filename_str = filename.numpy()
            image, kspace, contrast = from_multicoil_train_file_to_image_and_kspace_and_contrast(
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
        if parallel:
            kspace_size = (None, None)
        else:
            kspace_size = (None, None, None)
        image_size = (None, None)
        image.set_shape(n_slices + image_size)
        kspace.set_shape(n_slices + kspace_size)
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
        num_parallel_calls=tf.data.experimental.AUTOTUNE if rand or parallel else 2,
    )
    # contrast and size filtering
    if contrast or target_image_size is not None:
        image_and_kspace_and_contrast_ds = image_and_kspace_and_contrast_ds.filter(
            lambda image, kspace, tf_contrast: tf.logical_and(
                tf_contrast == contrast if contrast else True,
                tf.reduce_all(tf.less_equal(tf.shape(kspace)[-2:], target_image_size))
            )
        )
    image_and_kspace_ds = image_and_kspace_and_contrast_ds.map(
        lambda image, kspace, tf_contrast: (image, kspace),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if rand or parallel else 2,
    )
    if n_samples is not None:
        image_and_kspace_ds = image_and_kspace_ds.take(n_samples)
    masked_kspace_ds = image_and_kspace_ds.map(
        generic_from_kspace_to_masked_kspace_and_mask(
            AF=AF,
            scale_factor=scale_factor,
            parallel=parallel,
            fixed_masks=fixed_masks,
            output_shape_spec=output_shape_spec,
            mask_type=mask_type,
            batch_size=batch_size,
            target_image_size=target_image_size,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if rand or parallel else 2,
    )
    if batch_size is not None:
        masked_kspace_ds = masked_kspace_ds.batch(batch_size)
    masked_kspace_ds = masked_kspace_ds.repeat()
    if rand or parallel:
        masked_kspace_ds = masked_kspace_ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        masked_kspace_ds = masked_kspace_ds.prefetch(2)
    return masked_kspace_ds

def test_masked_kspace_dataset_from_indexable(
        path,
        AF=4,
        scale_factor=1,
        contrast=None,
        n_samples=None,
        output_shape_spec=False,
    ):
    r"""Dataset for the testing/challenge set of multi-coil fastMRI.

    The output of the dataset is of the form:
    ```
    undersampled_kspace, undersampling_mask, sensitivity_maps
    ```
    If `output_shape_spec` is True, the output shape specification is added.

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
        n_samples (int or None): the number of samples to consider for this inference.
            Useful for debugging purposes. If None, all samples are used.
            Defaults to None.
        output_shape_spec (bool): whether you need the output shape to be
            present in the model inputs. It is inferred from the ismrmrd header
            present in the file, via the encoding reconstruction space.
            Defaults to False.

    Returns:
        tf.data.Dataset: the testing/challenge dataset.
    """
    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', seed=0, shuffle=False)
    mask_and_kspace_and_contrast_and_image_size_ds = files_ds.map(
        tf_filename_to_mask_and_kspace_and_contrast_and_image_size,
    )
    # contrast filtering
    if contrast:
        mask_and_kspace_and_contrast_and_image_size_ds = mask_and_kspace_and_contrast_and_image_size_ds.filter(
            lambda mask, kspace, tf_contrast, image_size: tf_contrast == contrast
        )
    mask_and_kspace_and_image_size_ds = mask_and_kspace_and_contrast_and_image_size_ds.map(
        lambda mask, kspace, tf_contrast, image_size: (mask, kspace, image_size),
    )
    # af filtering
    if AF == 4:
        mask_and_kspace_and_image_size_ds = mask_and_kspace_and_image_size_ds.filter(
            lambda mask, kspace, image_size: tf_af(mask) < 5.5
        )
    else:
        mask_and_kspace_and_image_size_ds = mask_and_kspace_and_image_size_ds.filter(
            lambda mask, kspace, image_size: tf_af(mask) > 5.5
        )
    if n_samples is not None:
        mask_and_kspace_and_image_size_ds = mask_and_kspace_and_image_size_ds.take(n_samples)
    masked_kspace_ds = mask_and_kspace_and_image_size_ds.map(
        generic_prepare_mask_and_kspace(
            scale_factor=scale_factor,
            AF=AF,
            output_shape_spec=output_shape_spec,
        )
    )

    return masked_kspace_ds

def test_filenames(path, AF=4, contrast=None, n_samples=None):
    """The filenames associated with the test/challenge dataset function.

    This is useful when you need to know to which file each output of the
    `test_masked_kspace_dataset_from_indexable` corresponds to. You will typically
    use this when submitting to fastMRI.

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
        n_samples (int or None): the number of samples to consider for this inference.
            Useful for debugging purposes. If None, all samples are used.
            Defaults to None.

    Returns:
        tf.data.Dataset: the testing/challenge filenames dataset.
    """
    files_ds = tf.data.Dataset.list_files(f'{path}*.h5', seed=0, shuffle=False)
    mask_and_contrast_and_filename_ds = files_ds.map(
        tf_filename_to_mask_and_contrast_and_filename,
    )
    # contrast filtering
    if contrast:
        mask_and_contrast_and_filename_ds = mask_and_contrast_and_filename_ds.filter(
            lambda mask, tf_contrast, filename: tf_contrast == contrast
        )
    mask_and_filename_ds = mask_and_contrast_and_filename_ds.map(
        lambda mask, tf_contrast, filename: (mask, filename),
    )
    # af filtering
    if AF == 4:
        mask_and_filename_ds = mask_and_filename_ds.filter(
            lambda mask, filename: tf_af(mask) < 5.5
        )
    else:
        mask_and_filename_ds = mask_and_filename_ds.filter(
            lambda mask, filename: tf_af(mask) > 5.5
        )
    filename_ds = mask_and_filename_ds.map(
        lambda mask, filename: filename,
    )
    if n_samples is not None:
        filename_ds = filename_ds.take(n_samples)
    return filename_ds
