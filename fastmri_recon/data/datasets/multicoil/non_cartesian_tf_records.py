from functools import partial
from pathlib import Path

import tensorflow as tf

from fastmri_recon.data.utils.tfrecords import decode_ncmc_example
from fastmri_recon.data.utils.h5 import from_file_to_contrast


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
    if contrast is None and af == 4:
        filenames = sorted(list(Path(path).glob(f'*{pattern}.tfrecords')))
    else:
        filenames = get_tfrecords_files_for_contrast(path, contrast, pattern, af)
    raw_dataset = tf.data.TFRecordDataset(
        [str(f) for f in filenames],
        num_parallel_reads=2 if rand else None,
    ).apply(tf.data.experimental.ignore_errors(log_warning=True))
    if n_samples is not None:
        raw_dataset.take(n_samples)
    volume_ds = raw_dataset.map(
        partial(decode_ncmc_example, slice_random=rand, brain=brain),
        num_parallel_calls=2 if rand else None,
    )
    if rand:
        volume_ds = volume_ds.repeat().prefetch(buffer_size=2)
    return volume_ds

def get_tfrecords_files_for_contrast(path, contrast, pattern='radial', af=4):
    if af == 8:
        pattern = pattern + '_af8'
    tfrec_filenames = sorted(list(Path(path).glob(f'*{pattern}.tfrecords')))
    h5_filenames = sorted(list(Path(path).glob(f'*.h5')))
    assert len(tfrec_filenames) == len(h5_filenames)
    filtered_tfrec_filenames = [
        tfrec_filename for (tfrec_filename, h5_filename) in zip(tfrec_filenames, h5_filenames)
        if from_file_to_contrast(h5_filename) == contrast
    ]
    return filtered_tfrec_filenames
