import os

import tensorflow as tf

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as singlecoil_dataset
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet


def _extract_inputs_shape(inputs, no_batch=True):
    if isinstance(inputs, (list, tuple)):
        return [_extract_inputs_shape(i, no_batch=no_batch) for i in inputs]
    else:
        if no_batch:
            return [1] + inputs.shape[1:]
        else:
            return inputs.shape

def _extract_inputs_dtype(inputs):
    if isinstance(inputs, (list, tuple)):
        return [_extract_inputs_dtype(i) for i in inputs]
    else:
        return inputs.dtype

def _zeros_from_shape(shapes, dtypes):
    if isinstance(shapes, (list, tuple)):
        return [_zeros_from_shape(s, d) for s, d in zip(shapes, dtypes)]
    else:
        return tf.zeros(shapes, dtype=dtypes)

def evaluate_ncpdnet(
        multicoil=False,
        run_id='ncpdnet_sense_af4_1588609141',
        n_epochs=200,
        contrast=None,
        acq_type='radial',
        dcomp=False,
        n_iter=10,
        n_filters=32,
        n_primal=5,
        non_linearity='relu',
        n_samples=None,
        cuda_visible_devices='0123',
        **acq_kwargs,
    ):
    # this number means that 99.56% of all images will not be affected by
    # cropping
    im_size = (640, 400)
    if multicoil:
        val_path = f'{FASTMRI_DATA_DIR}multicoil_val/'
        raise ValueError('Non cartesian multicoil is not implemented yet')
    else:
        val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    run_params = {
        'n_primal': n_primal,
        'multicoil': multicoil,
        'activation': non_linearity,
        'n_iter': n_iter,
        'n_filters': n_filters,
        'im_size': im_size,
        'dcomp': dcomp,
    }

    if multicoil:
        pass
    else:
        dataset = singlecoil_dataset
        kwargs = acq_kwargs
    val_set = dataset(
        val_path,
        im_size,
        acq_type=acq_type,
        compute_dcomp=dcomp,
        contrast=contrast,
        inner_slices=None,
        rand=True,
        scale_factor=1e6,
        **kwargs
    )
    if n_samples is not None:
        val_set = val_set.take(n_samples)

    example_input = next(iter(val_set))[0]
    inputs_shape = _extract_inputs_shape(example_input, no_batch=True)
    inputs_dtype = _extract_inputs_dtype(example_input)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = NCPDNet(**run_params)
        inputs = _zeros_from_shape(inputs_shape, inputs_dtype)
        # special case for the shape:
        inputs[-1][0] = tf.constant([[372]])
        model(inputs)
        def tf_psnr(y_true, y_pred):
            perm_psnr = [3, 1, 2, 0]
            psnr = tf.image.psnr(
                tf.transpose(y_true, perm_psnr),
                tf.transpose(y_pred, perm_psnr),
                tf.reduce_max(y_true),
            )
            return psnr
        def tf_ssim(y_true, y_pred):
            perm_ssim = [0, 1, 2, 3]
            ssim = tf.image.ssim(
                tf.transpose(y_true, perm_ssim),
                tf.transpose(y_pred, perm_ssim),
                tf.reduce_max(y_true),
            )
            return ssim

        model.compile(loss=tf_psnr, metrics=[tf_ssim])
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    eval_res = model.evaluate(val_set, verbose=1, steps=199 if n_samples is None else None)
    return model.metrics_names, eval_res
