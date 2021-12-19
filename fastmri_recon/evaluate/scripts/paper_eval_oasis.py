from pathlib import Path
import random
import time
import warnings
warnings.filterwarnings("ignore")

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from fastmri_recon.config import OASIS_DATA_DIR, CHECKPOINTS_DIR
from fastmri_recon.data.sequences.oasis_sequences import Masked2DSequence, ZeroFilled2DSequence
from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from fastmri_recon.evaluate.reconstruction.zero_filled_reconstruction import reco_and_gt_zfilled_from_val_file
from fastmri_recon.evaluate.reconstruction.cross_domain_reconstruction import reco_and_gt_net_from_val_file
from fastmri_recon.evaluate.reconstruction.unet_reconstruction import reco_and_gt_unet_from_val_file
from fastmri_recon.models.functional_models.cascading import cascade_net
from fastmri_recon.models.functional_models.kiki import kiki_net
from fastmri_recon.models.functional_models.kiki_sep import full_kiki_net
from fastmri_recon.models.functional_models.pdnet import pdnet
from fastmri_recon.models.functional_models.old_unet import unet
from fastmri_recon.models.utils.non_linearities import lrelu


np.random.seed(0)

def evaluate_paper_oasis(n_samples=200):
    AF = 4
    # paths
    train_path = str(Path(OASIS_DATA_DIR) / 'OASIS_data') + '/'
    train_gen = Masked2DSequence(
        train_path,
        af=AF,
        inner_slices=32,
        scale_factor=1e-2,
        seed=0,
        rand=True,
        val_split=0.1,
    )
    val_gen_mask = train_gen.val_sequence
    n_train = 1000
    n_val = n_samples
    random.seed(0)
    train_gen.filenames = random.sample(train_gen.filenames, n_train)
    val_gen_mask.filenames = random.sample(val_gen_mask.filenames, n_val)

    train_gen_zero = ZeroFilled2DSequence(
        train_path,
        af=AF,
        inner_slices=32,
        scale_factor=1e-2,
        seed=0,
        rand=False,
        val_split=0.1,
        n_pooling=3,
    )
    val_gen_zero = train_gen_zero.val_sequence
    random.seed(0)
    train_gen_zero.filenames = random.sample(train_gen_zero.filenames, n_train)
    val_gen_zero.filenames = random.sample(val_gen_zero.filenames, n_val)

    all_net_params = [
        {
            'name': 'unet',
            'init_function': unet,
            'run_params': {
                'n_layers': 4,
                'pool': 'max',
                "layers_n_channels": [16, 32, 64, 128],
                'layers_n_non_lins': 2,
            },
            'val_gen': val_gen_zero,
            'run_id': 'UNet-OASIS',
        },
        {
            'name': 'pdnet',
            'init_function': pdnet,
            'run_params': {
                'n_primal': 5,
                'n_dual': 5,
                'n_iter': 10,
                'n_filters': 32,
            },
            'run_id': 'PDNet-OASIS',
        },
        {
            'name': 'cascadenet',
            'init_function': cascade_net,
            'run_params': {
                'n_cascade': 5,
                'n_convs': 5,
                'n_filters': 48,
                'noiseless': True,
            },
            'run_id': 'CascadeNet-OASIS',
        },
        {
            'name': 'kikinet-sep',
            'init_function': full_kiki_net,
            'run_params': {
                'n_convs': 16,
                'n_filters': 48,
                'noiseless': True,
                'activation': lrelu,
            },
            'run_id': 'KIKI-net-OASIS',
            'epoch': 50,
        },
    ]

    checkpoints_path = Path(CHECKPOINTS_DIR) / 'checkpoints'
    def unpack_model(init_function=None, run_params=None, run_id=None, epoch=300, **dummy_kwargs):
        try:
            model = init_function(input_size=(None, None, 1), fastmri=False, **run_params)
        except:
            model = init_function(input_size=(None, None, 1), **run_params)
        chkpt_path = checkpoints_path / f'{run_id}.hdf5'
        model.load_weights(str(chkpt_path))
        return model

    def metrics_for_params(val_gen=None, name=None, **net_params):
        if val_gen is None:
            val_gen = val_gen_mask
        model = unpack_model(**net_params)
        metrics = Metrics(METRIC_FUNCS)
        pred_and_gt = [
            reco_and_gt_net_from_val_file(*val_gen[i], model)
            for i in tqdm(range(len(val_gen)), desc=f'Val files for {name}')
        ]
        for im_recos, images in tqdm(pred_and_gt, desc=f'Stats for {name}'):
            metrics.push(images, im_recos)
        return metrics


    def metrics_zfilled():
        metrics = Metrics(METRIC_FUNCS)
        pred_and_gt = [
            reco_and_gt_zfilled_from_val_file(*val_gen_mask[i], crop=False)
            for i in tqdm(range(len(val_gen_mask)), desc='Val files for z-filled')
        ]
        for im_recos, images in tqdm(pred_and_gt, desc='Stats for z-filled'):
            metrics.push(images, im_recos)
        return metrics

    metrics = []
    for net_params in all_net_params:
        metrics.append((net_params['name'], metrics_for_params(**net_params)))

    metrics.append(('zfilled', metrics_zfilled()))

    metrics.sort(key=lambda x: x[1].metrics['PSNR'].mean())


    def n_model_params_for_params(reco_function=None, val_gen=None, name=None, **net_params):
        model = unpack_model(**net_params)
        n_params = model.count_params()
        return n_params
    n_params = {}
    for net_params in all_net_params:
        n_params[net_params['name']] = n_model_params_for_params(**net_params)

    n_params['zfilled'] = 0

    def runtime_for_params(val_gen=None, name=None, **net_params):
        if val_gen is None:
            val_gen = val_gen_mask
        model = unpack_model(**net_params)
        data = val_gen[0]
        start = time.time()
        reco_and_gt_net_from_val_file(*data, model)
        end = time.time()
        return end - start

    def runtime_zfilled():
        data = val_gen_mask[0]
        start = time.time()
        reco_and_gt_zfilled_from_val_file(*data, crop=False)
        end = time.time()
        return end - start

    runtimes = {}
    for net_params in tqdm(all_net_params):
        runtimes[net_params['name']] = runtime_for_params(**net_params)

    runtimes['zfilled'] = runtime_zfilled()

    metrics_table = pd.DataFrame(
        index=[name for name, _ in metrics],
        columns=['PSNR-mean (std) (dB)', 'SSIM-mean (std)', '# params', 'Runtime (s)'],
    )
    for name, m in metrics:
        metrics_table.loc[name, 'PSNR-mean (std) (dB)'] = "{mean:.4} ({std:.4})".format(
            mean=m.metrics['PSNR'].mean(),
            std=m.metrics['PSNR'].stddev(),
        )
        metrics_table.loc[name, 'SSIM-mean (std)'] = "{mean:.4} ({std:.4})".format(
            mean=m.metrics['SSIM'].mean(),
            std=m.metrics['SSIM'].stddev(),
        )
        metrics_table.loc[name, '# params'] = "{}".format(
            n_params[name],
        )
        metrics_table.loc[name, 'Runtime (s)'] = "{runtime:.4}".format(
            runtime=runtimes[name],
        )

    print(metrics_table)
    return metrics_table

@click.command()
@click.option(
    '-n',
    '--n-samples',
    default=200,
    type=int,
    help='The number of samples to use for the evaluation',
)
def evaluate_paper_oasis_click(n_samples):
    evaluate_paper_oasis(n_samples)


if __name__ == '__main__':
    evaluate_paper_oasis_click()
