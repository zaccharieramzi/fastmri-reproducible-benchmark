from functools import partial
from pathlib import Path

import tensorflow as tf

from fastmri_recon.data.utils.tfrecords import decode_ncmc_example


def train_nc_kspace_dataset_from_tfrecords(
        path,
        acq_type='radial',
        n_samples=None,
        slice_random=True,
    ):
    pattern = acq_type
    filenames = sorted(list(Path(path).glob(f'*{pattern}.tfrecords')))
    raw_dataset = tf.data.TFRecordDataset(
        [str(f) for f in filenames],
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )
    if n_samples is not None:
        raw_dataset.take(n_samples)
    volume_ds = raw_dataset.map(
        partial(decode_ncmc_example, slice_random=slice_random),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return volume_ds
