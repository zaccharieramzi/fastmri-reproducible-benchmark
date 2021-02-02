from pathlib import Path

import tensorflow as tf
from tfkbnufft.kbnufft import KbNufftModule
from tqdm import tqdm

from fastmri_recon.config import OASIS_DATA_DIR
from fastmri_recon.data.datasets.oasis_preprocessing import non_cartesian_from_volume_to_nc_kspace_and_traj
from fastmri_recon.data.utils.nii import from_file_to_volume
from fastmri_recon.data.utils.tfrecords import encode_example, get_extension_for_acq


def generate_oasis_tf_records(
        acq_type='radial_stacks',
        af=4,
        mode='train',
        shard=0,
        shard_size=3300,
        slice_size=176,
    ):
    tf.config.experimental_run_functions_eagerly(
        True,
    )
    path = Path(OASIS_DATA_DIR) / mode
    filenames = sorted(list(path.glob('*.nii.gz')))
    filenames = filenames[shard*shard_size:(shard+1)*shard_size]
    scale_factor = 1e-2
    volume_size = (slice_size, 256, 256)
    extension = get_extension_for_acq(
        volume_size,
        acq_type=acq_type,
        compute_dcomp=True,
        scale_factor=scale_factor,
        af=af,
    )
    extension = extension + '.tfrecords'
    nufft_ob = KbNufftModule(
        im_size=volume_size,
        grid_size=None,
        norm='ortho',
    )
    volume_transform = non_cartesian_from_volume_to_nc_kspace_and_traj(
        nufft_ob,
        volume_size,
        acq_type=acq_type,
        scale_factor=scale_factor,
        compute_dcomp=True,
        af=af,
    )
    for filename in tqdm(filenames):
        directory = filename.parent
        filename_tfrecord = directory / (filename.stem + extension)
        if filename_tfrecord.exists():
            continue
        volume = from_file_to_volume(filename)
        if volume.shape[0] % 2 != 0:
            continue
        if volume.shape[0] == 36 or volume.shape[0] == 44:
            continue
        with tf.device('/gpu:0'):
            volume = tf.constant(volume, dtype=tf.complex64)
            model_inputs, model_outputs = volume_transform(volume)
        with tf.io.TFRecordWriter(str(filename_tfrecord)) as writer:
            example = encode_example(model_inputs, model_outputs, compute_dcomp=True)
            writer.write(example)
