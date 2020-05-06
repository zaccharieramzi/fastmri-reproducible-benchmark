from dask.distributed import Client
from dask_jobqueue import SLURMCluster

def train_on_jz_dask(job_name, train_function, *args, **kwargs):
    cluster = SLURMCluster(
        cores=1,
        job_cpu=20,
        memory='80GB',
        job_name=job_name,
        walltime='60:00:00',
        interface='ib0',
        job_extra=[
            f'--gres=gpu:1',
            '--qos=qos_gpu-t4',
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
        train_function,
        *args,
        **kwargs,
        # this function has potential side effects
        pure=True,
    )
    client.gather(futures)
    print('Shutting down dask workers')
