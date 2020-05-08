import os

import click
import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import test_masked_kspace_dataset_from_indexable, test_filenames
from fastmri_recon.models.subclassed_models.updnet import UPDNet
from ..utils.write_results import write_result


def updnet_sense_inference(
        run_id='updnet_sense_af4_1588609141',
        exp_id='updnet',
        n_epochs=200,
        contrast=None,
        af=4,
        n_iter=10,
        n_layers=3,
        base_n_filter=16,
        non_linearity='relu',
        channel_attention_kwargs=None,
        n_samples=None,
        cuda_visible_devices='0123',
    ):
    test_path = f'{FASTMRI_DATA_DIR}multicoil_test_v2/'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    af = int(af)

    run_params = {
        'n_primal': 5,
        'n_dual': 1,
        'primal_only': True,
        'multicoil': True,
        'n_layers': n_layers,
        'layers_n_channels': [base_n_filter * 2**i for i in range(n_layers)],
        'non_linearity': non_linearity,
        'n_iter': n_iter,
        'channel_attention_kwargs': channel_attention_kwargs,
    }

    test_set = test_masked_kspace_dataset_from_indexable(
        test_path,
        AF=af,
        contrast=contrast,
        scale_factor=1e6,
        n_samples=n_samples,
    )
    test_set_filenames = test_filenames(
        test_path,
        AF=af,
        contrast=contrast,
        n_samples=n_samples,
    )

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = UPDNet(**run_params)
        model([
            tf.zeros([1, 15, 640, 372, 1], dtype=tf.complex64),
            tf.zeros([1, 15, 640, 372], dtype=tf.complex64),
            tf.zeros([1, 15, 640, 372], dtype=tf.complex64),
        ])
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs}.hdf5')
    tqdm_total = 199 if n_samples is None else None
    for data_example, filename in tqdm(zip(test_set, test_set_filenames), total=tqdm_total):
        res = model.predict(data_example)
        write_result(exp_id, res, filename)
