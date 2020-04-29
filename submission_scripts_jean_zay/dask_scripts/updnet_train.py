import click

from fastmri_recon.training_scripts.multi_coil.updnet_approach_sense import train_updnet
from generic_dask_training import train_on_jz_dask


@click.command()
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
    type=click.Choice(['CORPDFS_FBK', 'CORPD_FBK', None], case_sensitive=False),
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
def train_updnet_sense_dask(af, contrast, cuda_visible_devices, n_samples, n_epochs, n_iter, non_linearity):
    job_name = f'train_updnet_sense_{af}'
    if contrast is not None:
        job_name += f'_{contrast}'
    train_on_jz_dask(
        job_name,
        train_updnet,
        af, contrast, cuda_visible_devices, n_samples, n_epochs, n_iter,
        non_linearity=non_linearity,
    )


if __name__ == '__main__':
    train_updnet_sense_dask()
