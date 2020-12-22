from pathlib import Path

import tensorflow as tf

from fastmri_recon.data.utils.tfrecords import decode_postproc_example


def train_postproc_dataset_from_tfrecords(
        path,
        run_id,
        n_samples=None,
    ):
    extension = f'_{run_id}.tfrecords'
    filenames = sorted(list(Path(path).glob(f'*{extension}')))
    raw_dataset = tf.data.TFRecordDataset(
        [str(f) for f in filenames],
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )
    if n_samples is not None:
        raw_dataset.take(n_samples)
    volume_ds = raw_dataset.map(
        decode_postproc_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return volume_ds
