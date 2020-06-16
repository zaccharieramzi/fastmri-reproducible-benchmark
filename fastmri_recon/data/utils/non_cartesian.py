import numpy as np
import tensorflow as tf

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
