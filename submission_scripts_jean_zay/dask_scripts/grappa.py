from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from fastmri_recon.evaluate.scripts.grappa_eval import eval_grappa
from fastmri_recon.evaluate.scripts.grappa_inference import grappa_inference


def full_pipeline_dask():
    job_name = 'grappa'
    acceleration_factors = [4, 8]
    contrasts = ['CORPDFS_FBK', 'CORPD_FBK']
    # inference and eval
    inference_eval_cluster = SLURMCluster(
        cores=1,
        job_cpu=10,
        memory='20GB',
        job_name=job_name,
        walltime='1:00:00',
        interface='ib0',
        job_extra=[
            f'--gres=gpu:0',
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
    inference_futures = []
    eval_futures = []
    for af in acceleration_factors:
        for contrast in contrasts:
            inference_futures += [client.submit(
                # function to execute
                grappa_inference,
                contrast=contrast,
                af=af,
                exp_id=job_name,
                # this function has potential side effects
                pure=True,
            )]
            eval_futures += [client.submit(
                # function to execute
                eval_grappa,
                contrast=contrast,
                af=af,
                # this function has potential side effects
                pure=True,
            )]
    client.gather(inference_futures)
    # eval printing
    i = 0
    for af in acceleration_factors:
        for contrast in contrasts:
            m = client.gather(eval_futures[i])
            print('AF', af)
            print('Contrast', contrast)
            print(m)
            i += 1
    print('Shutting down dask workers')
    client.close()
    inference_eval_cluster.close()
