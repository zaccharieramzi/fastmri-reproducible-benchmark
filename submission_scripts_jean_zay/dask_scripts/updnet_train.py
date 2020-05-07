import click

from fastmri_recon.training_scripts.multi_coil.updnet_approach_sense import train_updnet
from generic_dask import train_on_jz_dask


@click.command()
@click.option(
    'original_run_id',
    '-id',
    default=None,
    type=str,
    help='The original run id for fine tuning. Defaults to None.',
)
@click.option(
    'af',
    '-a',
    default='4',
    type=click.Choice(['4', '8']),
    help='The acceleration factor chosen for this fine tuning. Defaults to 4.',
)
@click.option(
    'contrast',
    '-c',
    default=None,
    type=click.Choice(['CORPDFS_FBK', 'CORPD_FBK'], case_sensitive=False),
    help='The contrast chosen for this fine-tuning. Defaults to None.',
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
    'n_samples',
    '-ns',
    default=None,
    type=int,
    help='The number of samples to use for this training. Default to None, which means all samples are used.',
)
@click.option(
    'n_epochs',
    '-e',
    default=300,
    type=int,
    help='The number of epochs to train the model. Default to 300.',
)
@click.option(
    'n_iter',
    '-i',
    default=10,
    type=int,
    help='The number of epochs to train the model. Default to 300.',
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
    type=click.Choice(['dense', 'conv']),
    help='The type of channel attention to use. Default to None.',
)
@click.option(
    'loss',
    '-l',
    default='mae',
    type=click.Choice(['mae', 'mse', 'compound_mssim']),
    help='The loss on which to train. Default to mae.',
)
@click.option(
    'refine_smaps',
    '-rfs',
    is_flag=True,
    help='Whether you want to refine sensitivity maps using a trained unet.',
)
def train_updnet_sense_dask(
        original_run_id,
        af,
        contrast,
        cuda_visible_devices,
        n_samples,
        n_epochs,
        n_iter,
        non_linearity,
        n_layers,
        base_n_filter,
        channel_attention,
        loss,
        refine_smaps,
    ):
    job_name = f'train_updnet_sense_{af}'
    if contrast is not None:
        job_name += f'_{contrast}'
    if channel_attention == 'dense':
        channel_attention_kwargs = {'dense': True}
    elif channel_attention == 'conv':
        channel_attention_kwargs = {'dense': False}
    else:
        channel_attention_kwargs = None
    train_on_jz_dask(
        job_name,
        train_updnet,
        af, contrast, cuda_visible_devices, n_samples, n_epochs, n_iter,
        non_linearity=non_linearity,
        n_layers=n_layers,
        base_n_filter=base_n_filter,
        channel_attention_kwargs=channel_attention_kwargs,
        loss=loss,
        original_run_id=original_run_id,
        refine_smaps=refine_smaps,
    )


if __name__ == '__main__':
    train_updnet_sense_dask()
