from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import FASTMRI_DATA_DIR, CHECKPOINTS_DIR
from fastmri_recon.data.utils.h5 import from_multicoil_train_file_to_image_and_kspace_and_contrast
from fastmri_recon.data.utils.tfrecords import encode_postproc_example
from fastmri_recon.data.datasets.multicoil.preprocessing import generic_from_kspace_to_masked_kspace_and_mask
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet


def generate_postproc_tf_records(
        model_fun,
        model_kwargs,
        run_id,
        brain=False,
        n_epochs=200,
        af=4,
        n_iter=10,
        res=True,
        n_scales=0,
        n_primal=5,
        refine_smaps=False,
        refine_big=False,
        primal_only=True,
        n_dual=1,
        n_dual_filters=16,
        mode='train',
    ):
    main_path = Path(FASTMRI_DATA_DIR)
    if brain:
        path = main_path / f'brain_multicoil_{mode}'
    else:
        path = main_path / f'multicoil_{mode}'
    filenames = sorted(list(path.glob('*.h5')))
    kspace_transform = generic_from_kspace_to_masked_kspace_and_mask(
        AF=af,
        scale_factor=1e6,
        parallel=False,
        fixed_masks=False,
        output_shape_spec=brain,
        mask_type='equidistant_fake' if brain else 'random',
        batch_size=None,
        target_image_size=(640, 400),
    )
    class PreProcModel(tf.keras.models.Model):
        def call(self, inputs):
            image, kspace = inputs
            return kspace_transform(image, kspace)
    selection = [
        {'inner_slices': None, 'rand': False},  # slice selection
        {'rand': False, 'keep_dim': False},  # coil selection
    ]
    extension = f'_{run_id}.tfrecords'
    # Model init
    af = int(af)

    run_params = {
        'n_primal': n_primal,
        'multicoil': True,
        'n_scales': n_scales,
        'n_iter': n_iter,
        'refine_smaps': refine_smaps,
        'refine_big': refine_big,
        'res': res,
        'output_shape_spec': brain,
        'primal_only': primal_only,
        'n_dual': n_dual,
        'n_dual_filters': n_dual_filters,
    }
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        preproc_model = PreProcModel()
        model = XPDNet(model_fun, model_kwargs, **run_params)
        fake_inputs = [
            tf.zeros([1, 15, 640, 372, 1], dtype=tf.complex64),
            tf.zeros([1, 15, 640, 372], dtype=tf.complex64),
            tf.zeros([1, 15, 640, 372], dtype=tf.complex64),
        ]
        if brain:
            fake_inputs.append(tf.constant([[320, 320]]))
        model(fake_inputs)
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
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
        res = model.predict(model_inputs, batch_size=4)
        with tf.io.TFRecordWriter(str(filename_tfrecord)) as writer:
            example = encode_postproc_example([res], [model_outputs])
            writer.write(example)
