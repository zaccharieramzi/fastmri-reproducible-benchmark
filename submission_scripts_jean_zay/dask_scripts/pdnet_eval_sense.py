import click
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from fastmri_recon.evaluate.scripts.pdnet_sense_eval import evaluate_pdnet_sense


@click.command()
@click.option(
    'run_id',
    '-r',
    default='pdnet_sense_af4_1586266200',
    type=str,
    help='The run id of the trained network. Defaults to pdnet_sense_af4_1586266200.',
)
@click.option(
    'contrast',
    '-c',
    default=None,
    type=click.Choice(['CORPDFS_FBK', 'CORPD_FBK', None], case_sensitive=False),
    help='The contrast chosen for this evaluation. Defaults to CORPDFS_FBK.',
)
@click.option(
    'af',
    '-a',
    default='4',
    type=click.Choice(['4', '8']),
    help='The acceleration factor chosen for this fine tuning. Defaults to 4.',
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
def evaluate_pdnet_sense_dask(run_id, contrast, af, n_iter, cuda_visible_devices, n_samples):
    job_name = f'evaluate_pdnet_sense_{af}'
    if contrast is not None:
        job_name += f'_{contrast}'

    cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='160GB',
        job_name=job_name,
        walltime='20:00:00',
        interface='ib0',
        job_extra=[
            f'--gres=gpu:4',
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
        evaluate_pdnet_sense,
        # *args
        run_id, contrast, int(af), n_iter, n_samples, cuda_visible_devices,
        # this function has potential side effects
        pure=True,
    )
    metrics_names, eval_res = client.gather(futures)
    print(metrics_names)
    print(eval_res)
    print('Shutting down dask workers')


if __name__ == '__main__':
    evaluate_pdnet_sense_dask()
