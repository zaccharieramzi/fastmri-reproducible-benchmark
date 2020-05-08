import click

from fastmri_recon.evaluate.scripts.updnet_sense_inference import updnet_sense_inference
from generic_dask import infer_on_jz_dask


@click.command()
@click.argument(
    'runs',
    nargs=-1,
    type=str,
)
@click.option(
    'exp_id',
    '-x',
    default='updnet',
    type=str,
    help='The exp id of the experiment. Defaults to updnet.',
)
@click.option(
    'n_epochs',
    '-e',
    default=200,
    type=int,
    help='The number of epochs for which the model was trained or fine-tuned. Defaults to 200.',
)
@click.option(
    'n_iter',
    '-i',
    default=10,
    type=int,
    help='The number of epochs to train the model. Default to 300.',
)
@click.option(
    '-n',
    'n_samples',
    default=None,
    type=int,
    help='The number of samples to take from the dataset. Default to None (all samples taken).',
)
@click.option(
    'cuda_visible_devices',
    '-gpus',
    '--cuda-visible-devices',
    default='0123',
    type=str,
    help='The visible GPU devices. Defaults to 0123',
)
@click.option(
    'non_linearity',
    '-nl',
    default='relu',
    type=str,
    help='The non linearity to use in the model. Default to relu.',
)
@click.option(
    'n_layers',
    '-la',
    default=3,
    type=int,
    help='The number of layers in the u-net. Default to 3.',
)
@click.option(
    'base_n_filter',
    '-nf',
    default=16,
    type=int,
    help='The number of base filters in the u-net (x2 each layer). Default to 16.',
)
@click.option(
    'channel_attention',
    '-ca',
    default=None,
    type=click.Choice([None, 'dense', 'conv']),
    help='The type of channel attention to use. Default to None.',
)
def infer_updnet_sense_dask(
        runs,
        exp_id,
        n_epochs,
        n_iter,
        cuda_visible_devices,
        n_samples,
        non_linearity,
        n_layers,
        base_n_filter,
        channel_attention,
    ):
    job_name = f'eval_{exp_id}'
    n_runs = len(runs)
    if n_runs % 3 != 0:
        raise ValueError('You need to give the runs in triplets')
    runs_list = []
    for i in range(n_runs // 3):
        runs_list.append((
            runs[3*i],  # contrast
            runs[3*i + 1],  # af
            runs[3*i + 2],  # run id
        ))
    if channel_attention == 'dense':
        channel_attention_kwargs = {'dense': True}
    elif channel_attention == 'conv':
        channel_attention_kwargs = {'dense': False}
    else:
        channel_attention_kwargs = None

    infer_on_jz_dask(
        job_name,
        updnet_sense_inference,
        runs_list,
        exp_id=exp_id,
        cuda_visible_devices=cuda_visible_devices,
        n_samples=n_samples,
        n_epochs=n_epochs,
        n_iter=n_iter,
        non_linearity=non_linearity,
        n_layers=n_layers,
        base_n_filter=base_n_filter,
        channel_attention_kwargs=channel_attention_kwargs,
    )


if __name__ == '__main__':
    infer_updnet_sense_dask()
