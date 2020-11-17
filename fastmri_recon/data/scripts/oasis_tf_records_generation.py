from pathlib import Path

import tensorflow as tf

from fastmri_recon.config import OASIS_DATA_DIR
from fastmri_recon.data.datasets.oasis_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable as three_d_dataset
from fastmri_recon.data.utils.tfrecords import encode_example, get_extension_for_acq


def generate_oasis_tf_records(acq_type='radial_stacks', af=4, mode='train'):
    path = Path(OASIS_DATA_DIR) / mode
    scale_factor = 1e-2
    volume_size = (256, 256, 256)
    with tf.device('/gpu:0'):
        preprocessed_dataset = three_d_dataset(
            str(path),
            volume_size,
            acq_type=acq_type,
            compute_dcomp=True,
            scale_factor=scale_factor,
            af=af,
        )
    files_ds = tf.data.Dataset.list_files(f'{str(path)}*.nii.gz', shuffle=False)
    extension = get_extension_for_acq(
        volume_size,
        acq_type=acq_type,
        compute_dcomp=True,
        scale_factor=scale_factor,
        af=af,
    )
    extension = extension + '.tfrecords'
    for (model_inputs, model_outputs), filename in zip(preprocessed_dataset, files_ds):
        filename = Path(filename.numpy())
        directory = filename.parent
        filename_tfrecord = directory / filename.stem + extension
        with tf.io.TFRecordWriter(filename_tfrecord) as writer:
            example = encode_example(model_inputs, model_outputs, compute_dcomp=True)
            writer.write(example)
