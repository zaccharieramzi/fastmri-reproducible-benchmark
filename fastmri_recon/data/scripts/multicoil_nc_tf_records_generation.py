from pathlib import Path

import tensorflow as tf
from tfkbnufft.kbnufft import KbNufftModule
from tqdm import tqdm

from fastmri_recon.config import FASTMRI_DATA_DIR
from fastmri_recon.data.datasets.multicoil.preprocessing import non_cartesian_from_kspace_to_nc_kspace_and_traj
from fastmri_recon.data.utils.h5 import from_multicoil_train_file_to_image_and_kspace_and_contrast
from fastmri_recon.data.utils.tfrecords import encode_ncmc_example

def generate_multicoil_nc_tf_records(
        acq_type='radial',
        af=4,
        mode='train',
    ):
    path = Path(FASTMRI_DATA_DIR) / f'multicoil_{mode}'
    filenames = sorted(list(path.glob('*.h5')))
    scale_factor = 1e6
    image_size = (640, 400)
    nufft_ob = KbNufftModule(
        im_size=image_size,
        grid_size=None,
        norm='ortho',
    )
    transform = non_cartesian_from_kspace_to_nc_kspace_and_traj(
        nufft_ob,
        image_size,
        acq_type=acq_type,
        scale_factor=scale_factor,
        af=af,
        multiprocessing=False
    )
    class PreProcModel(tf.keras.models.Model):
        def call(self, inputs):
            image, kspace = inputs
            return transform(image, kspace)
    extension = f'_nc_{acq_type}.tfrecords'
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
        model_inputs, model_outputs = preproc_model.predict([image, kspace])
        with tf.io.TFRecordWriter(str(filename_tfrecord)) as writer:
            example = encode_ncmc_example(model_inputs, [model_outputs])
            writer.write(example)
