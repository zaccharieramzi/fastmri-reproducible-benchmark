import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_postproc_h5 import PostprocH5DatasetBuilder
from fastmri_recon.models.subclassed_models.post_processing_3d import PostProcessVnet
from fastmri_recon.evaluate.utils.write_results import write_result


def postproc_inference(
        recon_path,
        run_id,
        brain=False,
        n_epochs=200,
        contrast=None,
        af=4,
        scale_factor=1e6,
        exp_id='postproc',
        base_n_filters=16,
        n_scales=4,
        non_linearity='prelu',
    ):
    if brain:
        orig_path = f'{FASTMRI_DATA_DIR}brain_multicoil_test/'
    else:
        orig_path = f'{FASTMRI_DATA_DIR}multicoil_test_v2/'
    recon_path = orig_path + recon_path
    ds_builder = PostprocH5DatasetBuilder(orig_path, recon_path, af=af, contrast=contrast)
    run_params = dict(
        layers_n_channels=[base_n_filters*2**i for i in range(n_scales)],
        layers_n_non_lins=2,
        non_linearity=non_linearity,
        res=True,
    )
    model = PostProcessVnet(None, run_params)
    model(tf.zeros([2, 320, 320, 1]))
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    for recon, filename in tqdm(zip(ds_builder.raw_ds.as_numpy_iterator(), ds_builder.files_ds)):
        res = model.predict_batched(recon * scale_factor, batch_size=4)
        write_result(
            exp_id,
            res,
            filename.numpy().decode('utf-8'),
            scale_factor=1e6,
            brain=brain,
        )
