import tensorflow as tf
from tfkbnufft.kbnufft import KbNufftModule

from .oasis_preprocessing import non_cartesian_from_volume_to_nc_kspace_and_traj
from ..utils.nii import from_file_to_volume


def _tf_filename_to_volume(filename):
    def _from_train_file_to_volume_tensor_to_tensor(filename):
        filename_str = filename.numpy().decode("utf-8")
        volume = from_file_to_volume(
            filename_str,
        )
        return tf.convert_to_tensor(volume)
    [volume] = tf.py_function(
        _from_train_file_to_volume_tensor_to_tensor,
        [filename],
        [tf.complex64],
    )
    volume.set_shape((None, None, None))
    return volume

def train_nc_kspace_dataset_from_indexable(
        path,
        volume_size=(256, 256, 256),
        scale_factor=1,
        n_samples=None,
        acq_type='radial_stacks',
        compute_dcomp=False,
        shuffle=True,
        **acq_kwargs,
    ):
    files_ds = tf.data.Dataset.list_files(f'{path}*.nii.gz', shuffle=False)
    # this makes sure the file selection is happening once when using less than
    # all samples
    if shuffle:
        files_ds = files_ds.shuffle(
            buffer_size=1000,
            seed=0,
            reshuffle_each_iteration=False,
        )
    volume_ds = files_ds.map(
        _tf_filename_to_volume,
        num_parallel_calls=3,
    )
    # filter flat volumes and uneven
    volume_ds = volume_ds.filter(lambda x: tf.shape(x)[0] > 1)
    volume_ds = volume_ds.filter(lambda x: tf.math.mod(tf.shape(x)[0], 2) == 0)
    if n_samples is not None:
        volume_ds = volume_ds.take(n_samples)
    nufft_ob = KbNufftModule(
        im_size=volume_size,
        grid_size=None,
        norm='ortho',
    )
    volume_ds = volume_ds.map(
        non_cartesian_from_volume_to_nc_kspace_and_traj(
            nufft_ob,
            volume_size,
            acq_type=acq_type,
            scale_factor=scale_factor,
            compute_dcomp=compute_dcomp,
            **acq_kwargs,
        ),
        num_parallel_calls=3,
    ).repeat().prefetch(buffer_size=3)

    return volume_ds
