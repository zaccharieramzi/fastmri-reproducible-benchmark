import click

from fastmri_recon.evaluate.scripts.updnet_sense_eval import evaluate_updnet_sense
from generic_dask import eval_on_jz_dask


@click.command()
@click.option(
    'run_id',
    '-r',
    default='updnet_sense_af4_1588609141',
    type=str,
    help='The run id of the trained network. Defaults to updnet_sense_af4_1588609141.',
)
@click.option(
    'n_epochs',
    '-e',
    default=200,
    type=int,
    help='The number of epochs for which the model was trained or fine-tuned. Defaults to 200.',
)
@click.option(
    'contrast',
    '-c',
    default=None,
    type=click.Choice(['CORPDFS_FBK', 'CORPD_FBK',], case_sensitive=False),
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
@click.option(
    'refine_smaps',
    '-rfs',
    is_flag=True,
    help='Whether you want to refine sensitivity maps using a trained unet.',
)
def eval_updnet_sense_dask(
        run_id,
        n_epochs,
        contrast,
        af,
        n_iter,
        cuda_visible_devices,
        n_samples,
        non_linearity,
        n_layers,
        base_n_filter,
        channel_attention,
        refine_smaps,
    ):
    job_name = f'eval_updnet_sense_{af}'
    if contrast is not None:
        job_name += f'_{contrast}'
    if channel_attention == 'dense':
        channel_attention_kwargs = {'dense': True}
    elif channel_attention == 'conv':
        channel_attention_kwargs = {'dense': False}
    else:
        channel_attention_kwargs = None
    eval_on_jz_dask(
        job_name,
        evaluate_updnet_sense,
        af=af,
        contrast=contrast,
        cuda_visible_devices=cuda_visible_devices,
        n_samples=n_samples,
        n_epochs=n_epochs,
        n_iter=n_iter,
        non_linearity=non_linearity,
        n_layers=n_layers,
        base_n_filter=base_n_filter,
        channel_attention_kwargs=channel_attention_kwargs,
        refine_smaps=refine_smaps,
        run_id=run_id,
    )


if __name__ == '__main__':
    eval_updnet_sense_dask()
