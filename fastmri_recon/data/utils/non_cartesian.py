import numpy as np
import tensorflow as tf

from fastmri_recon.data.utils.masking.gen_mask_tf import gen_mask_tf


def get_radial_trajectory(nspokes, spokelength=None):
    def _get_radial_trajectory_numpy():
        ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
        kx = np.zeros(shape=(spokelength, nspokes))
        ky = np.zeros(shape=(spokelength, nspokes))
        ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
        for i in range(1, nspokes):
            kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
            ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

        ky = np.transpose(ky)
        kx = np.transpose(kx)

        traj = np.stack((ky.flatten(), kx.flatten()), axis=0)
        traj = tf.convert_to_tensor(traj, dtype=tf.float32)[None, ...]
        return traj
    traj = tf.py_function(
        _get_radial_trajectory_numpy,
        [],
        tf.float32,
    )
    traj.set_shape((1, 2, nspokes*spokelength))
    return traj

def get_cartesian_trajectory(af=4, readout_dim=400, spokelength=640):
    mask = gen_mask_tf(tf.zeros((1, spokelength, readout_dim)), accel_factor=af)
    y_taken = tf.cast(tf.where(mask)[:, 0], tf.int32)
    pi = tf.constant(np.pi)
    y_taken = (y_taken - (readout_dim/2)) * pi / (readout_dim/2)
    spoke_range = tf.range(spokelength)
    spoke_range = (spoke_range - (spokelength/2)) * pi / (spokelength/2)
    traj_readout, traj_spoke = tf.meshgrid(y_taken, spoke_range)
    traj = tf.stack([
        tf.reshape(traj_readout, [-1]),
        tf.reshape(traj_spoke, [-1]),
    ], axis=0)
    traj = traj[None, ...]

    return traj
