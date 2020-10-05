import os

import click
import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import test_masked_kspace_dataset_from_indexable, test_filenames
from fastmri_recon.models.subclassed_models.updnet import UPDNet
from fastmri_recon.evaluate.utils.write_results import write_result


def updnet_sense_inference(
        brain=False,
        challenge=False,
        run_id='updnet_sense_af4_1588609141',
        exp_id='updnet',
        n_epochs=200,
        contrast=None,
        scale_factor=1e6,
        af=4,
        n_iter=10,
        n_layers=3,
        base_n_filter=16,
        non_linearity='relu',
        channel_attention_kwargs=None,
        refine_smaps=False,
        n_samples=None,
        cuda_visible_devices='0123',
    ):
    if brain:
        if challenge:
            test_path = f'{FASTMRI_DATA_DIR}brain_multicoil_challenge/'
        else:
            test_path = f'{FASTMRI_DATA_DIR}brain_multicoil_test/'
    else:
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
        'refine_smaps': refine_smaps,
        'output_shape_spec': brain,
    }

    test_set = test_masked_kspace_dataset_from_indexable(
        test_path,
        AF=af,
        contrast=contrast,
        scale_factor=scale_factor,
        n_samples=n_samples,
        output_shape_spec=brain,
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
        fake_inputs = [
            tf.zeros([1, 15, 640, 372, 1], dtype=tf.complex64),
            tf.zeros([1, 15, 640, 372], dtype=tf.complex64),
            tf.zeros([1, 15, 640, 372], dtype=tf.complex64),
        ]
        if brain:
            fake_inputs.append(tf.constant([[320, 320]]))
        model(fake_inputs)
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    if n_samples is None:
        if not brain:
            if contrast:
                tqdm_total = n_volumes_test[af] // 2
            else:
                tqdm_total = n_volumes_test[af]
        else:
            if contrast:
                tqdm_total = brain_volumes_per_contrast['test'][af][contrast]
            else:
                tqdm_total = brain_n_volumes_test[af]
    else:
        tqdm_total = n_samples
    tqdm_desc = f'{exp_id}_{contrast}_{af}'

    # TODO: change when the following issue has been dealt with
    # https://github.com/tensorflow/tensorflow/issues/38561
    @tf.function(experimental_relax_shapes=True)
    def predict(t):
        return model(t)

    for data_example, filename in tqdm(zip(test_set, test_set_filenames), total=tqdm_total, desc=tqdm_desc):
        res = predict(data_example)
        write_result(
            exp_id,
            res.numpy(),
            filename.numpy().decode('utf-8'),
            scale_factor=scale_factor,
            brain=brain,
            challenge=challenge,
        )


@click.command()
@click.option(
    'af',
    '-a',
    type=int,
    default=4,
    help='The acceleration factor.'
)
@click.option(
    'brain',
    '-b',
    is_flag=True,
    help='Whether you want to consider brain data.'
)
@click.option(
    'challenge',
    '-ch',
    is_flag=True,
    help='Whether you want to consider challenge data (only for brain).'
)
@click.option(
    'n_iter',
    '-i',
    default=10,
    type=int,
    help='The number of epochs to train the model. Default to 300.',
)
@click.option(
    'refine_smaps',
    '-rfs',
    is_flag=True,
    help='Whether you want to use an smaps refiner.'
)
@click.option(
    'n_epochs',
    '-e',
    type=int,
    default=10,
    help='The number of epochs used in the final training.'
)
@click.option(
    'run_id',
    '-r',
    type=str,
    default=None,
    help='The run id of the final training.'
)
@click.option(
    'exp_id',
    '-x',
    type=str,
    default='updnet',
    help='The experiment id.'
)
@click.option(
    'contrast',
    '-c',
    type=str,
    default=None,
    help='The contrast to use for the training.'
)
def updnet_sense_inference_click(
        af,
        brain,
        challenge,
        n_iter,
        refine_smaps,
        n_epochs,
        run_id,
        exp_id,
        contrast,
    ):
    updnet_sense_inference(
        af=af,
        brain=brain,
        challenge=challenge,
        n_iter=n_iter,
        refine_smaps=refine_smaps,
        n_epochs=n_epochs,
        run_id=run_id,
        exp_id=exp_id,
        contrast=contrast,
    )


if __name__ == '__main__':
    updnet_sense_inference_click()
