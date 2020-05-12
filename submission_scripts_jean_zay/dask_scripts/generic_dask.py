from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from sklearn.model_selection import ParameterGrid

import os


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
    run_id = client.gather(futures)
    print(f'Train run id: {run_id}')
    print('Shutting down dask workers')

def eval_on_jz_dask(job_name, eval_function, *args, **kwargs):
    cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='80GB',
        job_name=job_name,
        walltime='20:00:00',
        interface='ib0',
        job_extra=[
            # for now we can't use 4 GPUs because of
            # https://github.com/tensorflow/tensorflow/issues/39268
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
        eval_function,
        *args,
        **kwargs,
        # this function has potential side effects
        pure=True,
    )
    metrics_names, eval_res = client.gather(futures)
    print(metrics_names)
    print(eval_res)
    print('Shutting down dask workers')

def infer_on_jz_dask(job_name, infer_function, runs, *args, **kwargs):
    cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='80GB',
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
    cluster.scale(len(runs))

    print(cluster.job_script())

    client = Client(cluster)
    futures = [client.submit(
        # function to execute
        infer_function,
        *args,
        contrast=contrast,
        af=af,
        run_id=run_id,
        **kwargs,
        # this function has potential side effects
        pure=True,
    ) for contrast, af, run_id in runs]
    client.gather(futures)
    print('Shutting down dask workers')

def full_pipeline_dask(job_name, train_function, eval_function, infer_function, **kwargs):
    # original training
    if os.environ.get('FASTMRI_DEBUG'):
        n_epochs_train = 1
        n_epochs_fine_tune = 1
        n_eval_samples = 1
        n_inference_samples = 1
    else:
        n_epochs_train = 200
        n_epochs_fine_tune = 50
        n_eval_samples = 50
        n_inference_samples = None
    train_cluster = SLURMCluster(
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
    train_cluster.scale(2)
    client = Client(train_cluster)
    acceleration_factors = [4, 8]
    futures = [client.submit(
        # function to execute
        train_function,
        af=af,
        n_epochs=n_epochs_train,
        **kwargs,
        # this function has potential side effects
        pure=True,
    ) for af in acceleration_factors]
    run_ids = client.gather(futures)
    train_cluster.scale(0)
    # fine tuning
    train_cluster.scale(4)
    contrasts = ['CORPDFS_FBK', 'CORPD_FBK']
    futures = []
    for af, run_id in zip(acceleration_factors, run_ids):
        for contrast in contrasts:
            futures += [client.submit(
                # function to execute
                train_function,
                af=af,
                contrast=contrast,
                original_run_id=run_id,
                n_epochs=n_epochs_fine_tune,
                **kwargs,
                # this function has potential side effects
                pure=True,
            ) for af in acceleration_factors]
    fine_tuned_run_ids = client.gather(futures)
    client.close()
    train_cluster.close()
    # inference and eval
    inference_eval_cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='80GB',
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
    inference_eval_cluster.scale(8)
    client = Client(inference_eval_cluster)
    i_run_id = 0
    inference_futures = []
    for af in acceleration_factors:
        for contrast in contrasts:
            run_id = fine_tuned_run_ids[i_run_id]
            inference_futures += [client.submit(
                # function to execute
                infer_function,
                contrast=contrast,
                af=af,
                run_id=run_id,
                n_epochs=n_epochs_fine_tune,
                n_samples=n_inference_samples,
                exp_id=job_name,
                **kwargs,
                # this function has potential side effects
                pure=True,
            )]
            eval_futures += [client.submit(
                # function to execute
                eval_function,
                contrast=contrast,
                af=af,
                run_id=run_id,
                n_epochs=n_epochs_fine_tune,
                n_samples=n_eval_samples,
                **kwargs,
                # this function has potential side effects
                pure=True,
            )]
            i_run_id += 1
    client.gather(inference_futures)
    # eval printing
    i_run_id = 0
    for af in acceleration_factors:
        for contrast in contrasts:
            metrics_names, eval_res = client.gather(eval_futures[i_run_id])
            print('AF', af)
            print('Contrast', contrast)
            print(metrics_names)
            print(eval_res)
    print('Shutting down dask workers')
    client.close()
    inference_eval_cluster.close()

def train_eval_parameter_grid(job_name, train_function, eval_function, parameter_grid):
    parameters = list(ParameterGrid(parameter_grid))
    n_parameters_config = len(parameters)
    train_cluster = SLURMCluster(
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
    train_cluster.scale(n_parameters_config)
    client = Client(train_cluster)
    futures = [client.submit(
        # function to execute
        train_function,
        **params,
    ) for params in parameters]
    run_ids = client.gather(futures)
    client.close()
    train_cluster.close()
    # eval
    eval_cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='80GB',
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
    eval_cluster.scale(n_parameters_config)
    client = Client(eval_cluster)
    for params in parameters:
        params.pop('n_samples')
    futures = [client.submit(
        # function to execute
        eval_function,
        run_id=run_id,
        n_samples=50,
        **params,
    ) for run_id, params in zip(run_ids, parameters)]

    for params, future in zip(parameters, futures):
        metrics_names, eval_res = client.gather(future)
        print('Parameters', params)
        print(metrics_names)
        print(eval_res)
    print('Shutting down dask workers')
    client.close()
    eval_cluster.close()
