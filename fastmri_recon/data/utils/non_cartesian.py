import numpy as np
from scipy.spatial.transform import Rotation as Rot
import tensorflow as tf

from fastmri_recon.data.utils.masking.gen_mask_tf import gen_mask_tf

def _rotation(ps, nstacks, nspokes):
    angle = np.pi / (nstacks * nspokes)
    n_rot = np.array((-np.pi - ps[-1]) * nstacks).astype(np.int64)
    p0 = np.cos(angle * n_rot) * ps[0] - np.sin(angle * n_rot) * ps[1]
    p1 = np.sin(angle * n_rot) * ps[0] + np.cos(angle * n_rot) * ps[1]
    return np.vstack([p0, p1, ps[-1]])

def _generate_stacks_for_traj(ktraj, nstacks, nspokes):
    n_measurements = ktraj.shape[-1]
    ktraj = np.tile(ktraj, [1, nstacks])
    z_locations = np.linspace(-0.5, 0.5, nstacks) * np.pi
    z_locations = np.repeat(z_locations, n_measurements)
    ktraj = np.concatenate([ktraj, z_locations[None, :]], axis=0)
    ktraj = _rotation(ktraj, nstacks, nspokes)
    return ktraj

# inspired by https://github.com/mmuckley/torchkbnufft/blob/master/profile_torchkbnufft.py#L144-L158
def get_radial_trajectory_numpy(nspokes, spokelength):
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
    return traj

def get_radial_trajectory(image_shape, af=None, us=None):
    if af is not None and us is not None:
        raise ValueError('You cannot set both acceleration and undersampling factor.')
    if af is None and us is None:
        raise ValueError('You need to set acceleration factor or undersampling factor.')
    spokelength = image_shape[-2]
    if af is not None:
        nspokes = image_shape[-1] // af
    if us is not None:
        nspokes = int(image_shape[-1] * np.pi / (2 * us))
    def _get_radial_trajectory_numpy():
        traj = get_radial_trajectory_numpy(nspokes, spokelength)
        traj = tf.convert_to_tensor(traj, dtype=tf.float32)[None, ...]
        return traj
    traj = tf.py_function(
        _get_radial_trajectory_numpy,
        [],
        tf.float32,
    )
    traj.set_shape((1, 2, nspokes*spokelength))
    return traj

def get_stacks_of_radial_trajectory(volume_shape, af=4):
    spokelength = volume_shape[-2]
    nspokes = volume_shape[-1] // af
    nstacks = volume_shape[0]
    def _get_stacks_of_radial_trajectory_numpy():
        ktraj = get_radial_trajectory_numpy(nspokes, spokelength)
        ktraj = _generate_stacks_for_traj(ktraj, nstacks, nspokes)

        traj = tf.convert_to_tensor(ktraj, dtype=tf.float32)[None, ...]
        return traj
    traj = tf.py_function(
        _get_stacks_of_radial_trajectory_numpy,
        [],
        tf.float32,
    )
    traj.set_shape((1, 3, nspokes*spokelength*nstacks))
    return traj


# spiral trajectory was inspired by internal work by @chaithyagr
def _complex_to_2d(points):
    X = points.real
    Y = points.imag
    return np.asarray([X, Y]).T

def get_spiral_trajectory_numpy(nspokes, spokelength, num_revolutions=3):
    shot = np.arange(0, spokelength // 2, dtype=np.complex)
    radius = shot / (spokelength//2) * np.pi
    angle = np.exp(2 * 1j * np.pi * shot / (spokelength//2) * num_revolutions)
    single_shot = np.multiply(radius, angle)
    single_shot = np.append(np.flip(single_shot, axis=0), -single_shot[1:])
    k_shots = []
    for i in np.arange(0, nspokes):
        shot_rotated = single_shot * np.exp(1j * 2 * np.pi * i / (nspokes * 2))
        k_shots.append(_complex_to_2d(shot_rotated))
    k_shots = np.asarray(k_shots)
    traj = np.transpose(k_shots, axes=(2, 1, 0))
    traj = traj.reshape([2, -1])
    return traj

def get_spiral_trajectory(image_shape, af=None, us=None, num_revolutions=3):
    spokelength = image_shape[-2]
    if af is not None and us is not None:
        raise ValueError('You cannot set both acceleration and undersampling factor.')
    if af is None and us is None:
        raise ValueError('You need to set acceleration factor or undersampling factor.')
    spokelength = image_shape[-2]
    if af is not None:
        nspokes = image_shape[-1] // af
    if us is not None:
        theta_max = 2 * np.pi * num_revolutions
        nspokes = int(image_shape[-1] * np.pi / (2 * us * theta_max))
    def _get_spiral_trajectory():
        traj = get_spiral_trajectory_numpy(nspokes, spokelength, num_revolutions=num_revolutions)
        traj = tf.convert_to_tensor(traj, dtype=tf.float32)[None, ...]
        return traj
    traj = tf.py_function(
        _get_spiral_trajectory,
        [],
        tf.float32,
    )
    traj.set_shape((1, 2, nspokes*(spokelength-1)))
    return traj

def get_stacks_of_spiral_trajectory(volume_shape, af=4, num_revolutions=3):
    spokelength = volume_shape[-2]
    nspokes = volume_shape[-1] // af
    nstacks = volume_shape[0]
    def _get_stacks_of_spiral_trajectory_numpy():
        ktraj = get_spiral_trajectory_numpy(nspokes, spokelength, num_revolutions=num_revolutions)
        ktraj = _generate_stacks_for_traj(ktraj, nstacks, nspokes)
        traj = tf.convert_to_tensor(ktraj, dtype=tf.float32)[None, ...]
        return traj
    traj = tf.py_function(
        _get_stacks_of_spiral_trajectory_numpy,
        [],
        tf.float32,
    )
    traj.set_shape((1, 3, nspokes*(spokelength-1)*nstacks))
    return traj

# based on internal work by chaithyagr
def get_3d_radial_trajectory_numpy(volume_shape, af=4):
    spokelength = volume_shape[-2]
    nshots = (volume_shape[-1] // af) * volume_shape[0]
    traj = _init_radial_trajectories(
        spokelength,
        int(nshots**(0.5)),
        dimension=3,
        initialization='RadialIO',
    )
    traj = traj.reshape([3, -1])
    traj = traj * (2 * np.pi) * np.pi
    return traj

def get_3d_radial_trajectory(volume_shape, af=4):
    spokelength = volume_shape[-2]
    nshots = (volume_shape[-1] // af) * volume_shape[0]
    nshots = int(nshots**(0.5))**2
    def _get_stacks_of_spiral_trajectory_numpy():
        ktraj = get_3d_radial_trajectory_numpy(volume_shape, af=af)
        traj = tf.convert_to_tensor(ktraj, dtype=tf.float32)[None, ...]
        return traj
    traj = tf.py_function(
        _get_stacks_of_spiral_trajectory_numpy,
        [],
        tf.float32,
    )
    traj.set_shape((1, 3, nshots*spokelength))
    return traj

def _init_radial_trajectories(num_samples, num_shots, dimension,
                              initialization, num_shots_z=None, **kwargs):
    k_shots = []
    shot = np.arange(0, num_samples, dtype=np.complex)
    if initialization == 'RadialIO':
        k_TE = num_samples // 2
        shot = (-1 * shot / k_TE + 1) * \
               (1 / (2 * np.pi)) * (1 - np.finfo(float).eps)
        theta = np.pi / num_shots
    else:
        shot = shot / (num_samples - 1) * \
               1 / (2 * np.pi) * (1 - np.finfo(float).eps)
        theta = 2 * np.pi / num_shots
        # Rotate shot in xy plane
    for k in np.arange(num_shots):
        angle_xy = k * theta + theta / 2
        rotated_shot = shot * np.exp(1j * (angle_xy))
        if dimension == 3:
            if num_shots_z is None:
                num_shots_z = num_shots
            shot2d = _complex_to_2d(rotated_shot)
            shot3d = np.zeros(
                (*shot2d.shape[0:-1], shot2d.shape[-1] + 1)
            )
            shot3d[:, 0:2] = shot2d
            for j in np.arange(num_shots_z):
                angle_z = theta / 2 + j * theta
                C = np.cos(angle_z / 2)
                S = np.sin(angle_z / 2)
                r = Rot.from_quat(
                    [C, - np.sin(angle_xy) * S,
                     np.cos(angle_xy) * S, 0])
                k_shots.append(r.apply(shot3d))
        else:
            k_shots.append(_complex_to_2d(rotated_shot))
    return np.asarray(k_shots)

def get_debugging_cartesian_trajectory():
    # we fix those to have a determined tensor shape
    af = 4
    readout_dim = 400
    spokelength = 640
    mask = gen_mask_tf(tf.zeros((1, spokelength, readout_dim)), accel_factor=af, fixed_masks=True)
    y_taken = tf.cast(tf.where(mask)[:, -1], tf.float32)
    pi = tf.constant(np.pi, dtype=tf.float32)
    y_taken = (y_taken - (readout_dim/2)) * pi / (readout_dim/2)
    spoke_range = tf.range(spokelength, dtype=tf.float32)
    spoke_range = (spoke_range - (spokelength/2)) * pi / (spokelength/2)
    traj_readout, traj_spoke = tf.meshgrid(y_taken, spoke_range)
    traj = tf.stack([
        tf.reshape(traj_readout, [-1]),
        tf.reshape(traj_spoke, [-1]),
    ], axis=0)
    traj = traj[None, ...]
    # the only reason why this is debugging is because we need a fixed number
    # of points for the trajectory to use in the data processing pipeline
    traj.set_shape((1, 2, 62080))
    return traj
