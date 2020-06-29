from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import pandas as pd

from fastmri_recon.evaluate.scripts.xpdnet_eval import evaluate_xpdnet
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.training_scripts.xpdnet_train import train_xpdnet

def train_eval_plug_and_play(
        contrast='CORPD_FBK',
        n_epochs=200,
        n_samples=None,
        af=4,
        n_primal=5,
        loss='compound_mssim',
        train_partition='gpu_p1',
        model_name=None,
        model_size=None,
    ):
    job_name = 'plug_and_play'
    model_specs = list(get_model_specs(force_res=False, n_primal=n_primal))
    if model_name is not None:
        model_specs = [ms for ms in model_specs if ms[0] == model_name]
    if model_size is not None:
        model_specs = [ms for ms in model_specs if ms[1] == model_size]
    n_models = len(model_specs)
    train_cluster = SLURMCluster(
        cores=1,
        job_cpu=20,
        memory='80GB',
        job_name=job_name,
        walltime='60:00:00',
        interface='ib0',
        job_extra=[
            '--gres=gpu:1',
            '--qos=qos_gpu-t4',
            '--distribution=block:block',
            '--hint=nomultithread',
            '--output=%x_%j.out',
            f'--partition {train_partition}',
        ],
        env_extra=[
            'cd $WORK/fastmri-reproducible-benchmark',
            '. ./submission_scripts_jean_zay/env_config.sh',
        ],
    )
    train_cluster.adapt(minimum_jobs=0, maximum_jobs=n_models)
    client = Client(train_cluster)
    futures = [client.submit(
        # function to execute
        train_xpdnet,
        model_fun=model_fun,
        model_kwargs=kwargs,
        model_size=model_size,
        multicoil=False,
        n_scales=n_scales,
        res=res,
        n_primal=n_primal,
        contrast=contrast,
        n_epochs=n_epochs,
        n_samples=n_samples,
        af=af,
        loss=loss,
    ) for _, model_size, model_fun, kwargs, _, n_scales, res in model_specs]
    run_ids = client.gather(futures)
    client.close()
    train_cluster.close()
    # eval
    eval_plug_and_play(
        run_ids,
        job_name=job_name,
        contrast=contrast,
        n_epochs=n_epochs,
        af=af,
        n_primal=n_primal,
        model_name=model_name,
        model_size=model_size,
        n_samples_train=n_samples,
    )
    return run_ids


def eval_plug_and_play(
        run_ids,
        job_name='eval_pandp',
        contrast='CORPD_FBK',
        n_samples_train=None,
        n_epochs=200,
        af=4,
        n_primal=5,
        train_partition='gpu_p1',
        model_name=None,
        model_size=None,
    ):
    model_specs = list(get_model_specs(force_res=False, n_primal=n_primal))
    if model_name is not None:
        model_specs = [ms for ms in model_specs if ms[0] == model_name]
    if model_size is not None:
        model_specs = [ms for ms in model_specs if ms[1] == model_size]
    n_models = len(model_specs)
    # eval
    eval_cluster = SLURMCluster(
        cores=1,
        job_cpu=40,
        memory='80GB',
        job_name=job_name,
        walltime='2:00:00',
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
    eval_cluster.adapt(minimum_jobs=0, maximum_jobs=n_models)
    client = Client(eval_cluster)

    futures = [client.submit(
        # function to execute
        evaluate_xpdnet,
        model_fun=model_fun,
        model_kwargs=kwargs,
        run_id=run_id,
        multicoil=False,
        n_samples=50,
        contrast=contrast,
        af=af,
        n_epochs=n_epochs,
        n_scales=n_scales,
        res=res,
    ) for run_id, (_, _, model_fun, kwargs, _, n_scales, res) in zip(run_ids, model_specs)]

    df_results = pd.DataFrame(columns='model_name model_size psnr ssim'.split())

    for (name, model_size, _, _, _, _, _), future in zip(model_specs, futures):
        _, eval_res = client.gather(future)
        df_results = df_results.append(dict(
            model_name=name,
            model_size=model_size,
            psnr=eval_res[0],
            ssim=eval_res[1],
        ), ignore_index=True)

    print(df_results)
    outputs_file = f'reconstruction_results_{n_samples_train}.csv'
    if model_name is not None:
        outputs_file = f'reconstruction_results_{n_samples_train}_{model_name}.csv'
    df_results.to_csv(outputs_file)
    print('Shutting down dask workers')
    client.close()
    eval_cluster.close()
    return run_ids
