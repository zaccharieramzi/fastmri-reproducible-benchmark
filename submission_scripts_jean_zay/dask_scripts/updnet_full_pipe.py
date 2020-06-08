import click

from fastmri_recon.evaluate.scripts.updnet_sense_eval import evaluate_updnet_sense
from fastmri_recon.evaluate.scripts.updnet_sense_inference import updnet_sense_inference
from fastmri_recon.training_scripts.updnet_train import train_updnet

from generic_dask import full_pipeline_dask


@click.command()
@click.argument(
    'exp_id',
    type=str,
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
@click.option(
    'single_coil',
    '-sc',
    is_flag=True,
    help='Whether you want to use single coil data.',
)
def full_pipe_updnet_sense_dask(
        exp_id,
        n_iter,
        non_linearity,
        n_layers,
        base_n_filter,
        channel_attention,
        loss,
        refine_smaps,
        single_coil,
    ):
    job_name = f'updnet_sense_{exp_id}'
    if channel_attention == 'dense':
        channel_attention_kwargs = {'dense': True}
    elif channel_attention == 'conv':
        channel_attention_kwargs = {'dense': False}
    else:
        channel_attention_kwargs = None
    full_pipeline_dask(
        job_name,
        train_function=train_updnet,
        eval_function=evaluate_updnet_sense,
        infer_function=updnet_sense_inference,
        n_iter=n_iter,
        non_linearity=non_linearity,
        n_layers=n_layers,
        base_n_filter=base_n_filter,
        channel_attention_kwargs=channel_attention_kwargs,
        loss=loss,
        refine_smaps=refine_smaps,
        multicoil=not single_coil,
    )


if __name__ == '__main__':
    full_pipe_updnet_sense_dask()
