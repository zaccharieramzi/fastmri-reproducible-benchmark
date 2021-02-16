from functools import partial
from pathlib import Path

import tensorflow as tf

from fastmri_recon.data.utils.tfrecords import decode_ncmc_example


def train_nc_kspace_dataset_from_tfrecords(
        path,
        image_size=(640, 400),
        acq_type='radial',
        n_samples=None,
        rand=True,
        scale_factor=1e6,
        compute_dcomp=True,
        af=4,
        contrast=None,
        inner_slices=None,
        brain=False,
    ):
    pattern = acq_type
    filenames = sorted(list(Path(path).glob(f'*{pattern}.tfrecords')))
    raw_dataset = tf.data.TFRecordDataset(
        [str(f) for f in filenames],
        num_parallel_reads=2,
    )
    if n_samples is not None:
        raw_dataset.take(n_samples)
    volume_ds = raw_dataset.map(
        partial(decode_ncmc_example, slice_random=rand, brain=brain),
        num_parallel_calls=2,
    ).repeat().prefetch(buffer_size=2)
    return volume_ds
