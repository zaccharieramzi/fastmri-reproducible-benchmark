from functools import partial
from pathlib import Path

import tensorflow as tf

from fastmri_recon.data.utils.tfrecords import decode_example, get_extension_for_acq


def train_nc_kspace_dataset_from_tfrecords(
        path,
        volume_size=(256, 256, 256),
        scale_factor=1,
        acq_type='radial_stacks',
        compute_dcomp=False,
        n_samples=None,
        **acq_kwargs,
    ):
    pattern = get_extension_for_acq(
        volume_size=volume_size,
        scale_factor=scale_factor,
        acq_type=acq_type,
        **acq_kwargs,
    )
    filenames = sorted(list(Path(path).glob(f'*{pattern}.tfrecords')))
    raw_dataset = tf.data.TFRecordDataset(
        [str(f) for f in filenames],
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )
    if n_samples is not None:
        raw_dataset.take(n_samples)
    volume_ds = raw_dataset.map(
        partial(decode_example, compute_dcomp=compute_dcomp),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return volume_ds
