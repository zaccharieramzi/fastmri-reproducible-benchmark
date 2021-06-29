from pathlib import Path

import tensorflow as tf
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.mri.dcomp_calc import calculate_density_compensator
from tqdm import tqdm

from fastmri_recon.config import FASTMRI_DATA_DIR
from fastmri_recon.data.utils.crop import adjust_image_size
from fastmri_recon.data.utils.fourier import tf_ortho_ifft2d
from fastmri_recon.data.utils.non_cartesian import get_radial_trajectory, get_debugging_cartesian_trajectory, get_spiral_trajectory
from fastmri_recon.data.utils.multicoil.smap_extract import extract_smaps, non_cartesian_extract_smaps
from fastmri_recon.data.utils.h5 import from_multicoil_train_file_to_image_and_kspace_and_contrast
from fastmri_recon.data.utils.tfrecords import encode_ncmc_example
from fastmri_recon.models.utils.fourier import tf_unmasked_adj_op, tf_unmasked_adj_op, nufft, FFTBase


def generate_multicoil_nc_tf_records(
        acq_type='radial',
        af=4,
        mode='train',
        brain=False,
    ):
    if brain:
        path = Path(FASTMRI_DATA_DIR) / f'brain_multicoil_{mode}'
    else:
        path = Path(FASTMRI_DATA_DIR) / f'multicoil_{mode}'
    filenames = sorted(list(path.glob('*.h5')))
    scale_factor = 1e6
    image_size = (640, 400)
    nufft_ob = KbNufftModule(
        im_size=image_size,
        grid_size=None,
        norm='ortho',
    )
    class PreProcModel(tf.keras.models.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            interpob = nufft_ob._extract_nufft_interpob()
            self.nufftob_back = kbnufft_adjoint(interpob, multiprocessing=False)
            self.nufftob_forw = kbnufft_forward(interpob, multiprocessing=False)
            if acq_type == 'radial':
                self.traj = get_radial_trajectory(image_size, af=af)
            elif acq_type == 'cartesian':
                self.traj = get_debugging_cartesian_trajectory()
            elif acq_type == 'spiral':
                self.traj = get_spiral_trajectory(image_size, af=af)
            else:
                raise NotImplementedError(f'{acq_type} dataset not implemented yet.')
            self.dcomp = calculate_density_compensator(
                interpob,
                self.nufftob_forw,
                self.nufftob_back,
                self.traj[0],
            )
            if brain:
                self.fft = FFTBase(False, multicoil=True, use_smaps=False)
        def call(self, inputs):
            images = inputs['image']
            kspaces = inputs['kspace']
            if brain:
                complex_images = self.fft.adj_op([kspaces[..., None], None])[..., 0]
                complex_images_padded = adjust_image_size(
                    complex_images,
                    image_size,
                    multicoil=True,
                )
                kspaces = self.fft.op([complex_images_padded[..., None], None])[..., 0]
            traj = tf.repeat(self.traj, tf.shape(images)[0], axis=0)
            orig_image_channels = tf_ortho_ifft2d(kspaces)
            nc_kspace = nufft(nufft_ob, orig_image_channels, traj, image_size, multiprocessing=False)
            nc_kspace_scaled = nc_kspace * scale_factor
            images_scaled = images * scale_factor
            images_channeled = images_scaled[..., None]
            nc_kspaces_channeled = nc_kspace_scaled[..., None]
            orig_shape = tf.ones([tf.shape(kspaces)[0]], dtype=tf.int32) * tf.shape(kspaces)[-1]
            dcomp = tf.ones([tf.shape(kspaces)[0], tf.shape(self.dcomp)[0]], dtype=self.dcomp.dtype) * self.dcomp[None, :]
            extra_args = (orig_shape, dcomp)
            smaps = non_cartesian_extract_smaps(nc_kspace, traj, dcomp, self.nufftob_back, orig_shape)
            model_inputs = (nc_kspaces_channeled, traj, smaps, extra_args)
            if brain:
                output_shape = tf.shape(images)[1:][None, :]
                output_shape = tf.tile(output_shape, [tf.shape(images)[0], 1])
                model_inputs += (output_shape,)
            return model_inputs, images_channeled

    extension = f'_nc_{acq_type}'
    if af != 4:
        extension += f'_af{af}'
    extension += '.tfrecords'
    selection = [
        {'inner_slices': None, 'rand': False},  # slice selection
        {'rand': False, 'keep_dim': False},  # coil selection
    ]
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        preproc_model = PreProcModel()
    for filename in tqdm(filenames):
        directory = filename.parent
        filename_tfrecord = directory / (filename.stem + extension)
        if filename_tfrecord.exists():
            continue
        image, kspace, _ = from_multicoil_train_file_to_image_and_kspace_and_contrast(
            filename,
            selection=selection,
        )
        data = tf.data.Dataset.zip({
            'image': tf.data.Dataset.from_tensor_slices(image),
            'kspace': tf.data.Dataset.from_tensor_slices(kspace),
        })
        data = data.batch(len(image))
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        data = data.with_options(options)
        model_inputs, model_outputs = preproc_model.predict(data)
        with tf.io.TFRecordWriter(str(filename_tfrecord)) as writer:
            example = encode_ncmc_example(model_inputs, [model_outputs])
            writer.write(example)
