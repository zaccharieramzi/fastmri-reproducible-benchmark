"""Script to download the model checkpoints from the HuggingFace Hub
"""
from pathlib import Path

import click
from huggingface_hub import hf_hub_download

from fastmri_recon.config import CHECKPOINTS_DIR


REPO_ID_BASE = 'zaccharieramzi/{model_name}-{dataset}'
MODEL_NAMES = ['UNet', 'KIKI-net', 'CascadeNet', 'PDNet']

def download_checkpoints(dataset=None):
    """Downloads the checkpoints from the HuggingFace Hub for the specified
    dataset

    Args:
        - dataset (str): one of 'fastmri' or 'OASIS'. Defaults to None,
        which means that both are downloaded.
    """
    if dataset is None:
        for d in ['fastmri', 'OASIS']:
            download_checkpoints(dataset=d)
    else:
        for model_name in MODEL_NAMES:
            repo_id = REPO_ID_BASE.format(model_name=model_name, dataset=dataset)
            hf_hub_download(
                repo_id=repo_id,
                filename='model_weights.h5',
                cache_dir=Path(CHECKPOINTS_DIR) / 'checkpoints',
                force_filename=f'{model_name}-{dataset}.hdf5',
            )

@click.command()
# the next option has a choice between fastmri and OASIS
@click.option(
    '-d',
    '--dataset',
    default=None,
    type=click.Choice(['fastmri', 'OASIS'], case_sensitive=True),
    help='The dataset to download the checkpoints for. Defaults to all.',
)
def download_checkpoints_cli(dataset):
    download_checkpoints(dataset=dataset)


if __name__ == '__main__':
    download_checkpoints_cli()