import click
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from fastmri_recon.training_scripts.multicoil.updnet_approach_sense import train_updnet


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
def train_updnet_sense_dask(af, contrast, cuda_visible_devices, n_samples, n_epochs, n_iter):
    job_name = f'train_updnet_sense_{af}'
    if contrast is not None:
        job_name += f'_{contrast}'

    cluster = SLURMCluster(
        cores=1,
        job_cpu=20,
        memory='80GB',
        job_name=job_name,
        walltime='20:00:00',
        interface='ib0',
        job_extra=[
            f'--gres=gpu:1',
            '--qos=qos_gpu-t3',
            '--distribution=block:block',
            '--hint=nomultithread',
            '--output=%x_%j.out',
        ],
        env_extra=[
            'cd $WORK/fastmri-reproducible-benchmark',
            '. ./submission_scripts_jean_zay/env_config.sh',
        ],
    )
    cluster.scale(1)

    print(cluster.job_script())

    client = Client(cluster)
    futures = client.submit(
        # function to execute
        train_updnet,
        # *args
        af, contrast, cuda_visible_devices, n_samples, n_epochs, n_iter,
        # this function has potential side effects
        pure=True,
    )
    client.gather(futures)
    print('Shutting down dask workers')


if __name__ == '__main__':
    train_updnet_sense_dask()
