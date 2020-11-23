from pathlib import Path
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

from fastmri_recon.config import FASTMRI_DATA_DIR
from fastmri_recon.data.sequences.fastmri_sequences import ZeroFilled2DSequence, Masked2DSequence
from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from fastmri_recon.evaluate.reconstruction.zero_filled_reconstruction import reco_and_gt_zfilled_from_val_file
from fastmri_recon.evaluate.reconstruction.cross_domain_reconstruction import reco_and_gt_net_from_val_file
from fastmri_recon.evaluate.reconstruction.unet_reconstruction import reco_and_gt_unet_from_val_file
from fastmri_recon.models.functional_models.cascading import cascade_net
from fastmri_recon.models.functional_models.kiki_sep import full_kiki_net
from fastmri_recon.models.functional_models.pdnet import pdnet
from fastmri_recon.models.functional_models.unet import unet
from fastmri_recon.models.utils.non_linearities import lrelu

np.random.seed(0)


plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['image.cmap'] = 'gray'

def evaluate_paper(AF=4, contrast=None, n_samples=None):
    if AF not in [4, 8]:
        raise ValueError(f'AF {AF} not correct.')
    if contrast not in [None, 'CORPD_FBK', 'CORPDFS_FBK']:
        raise ValueError(f'Contrast {contrast} is not correct.')


    val_path = Path(FASTMRI_DATA_DIR)/ 'singlecoil_val'
    val_gen_zero = ZeroFilled2DSequence(
        val_path,
        af=AF,
        norm=True,
        mode='validation',
        contrast=contrast,
    )
    val_gen_scaled = Masked2DSequence(
        val_path,
        mode='validation',
        af=AF,
        scale_factor=1e6,
        contrast=contrast,
    )
    if n_samples is not None:
        val_gen_zero.filenames = val_gen_zero.filenames[:n_samples]
        val_gen_scaled.filenames = val_gen_scaled.filenames[:n_samples]

    # TODO: get the correct run ids in function of the AF
    all_net_params = [
        {
            'name': 'unet',
            'init_function': unet,
            'run_params': {
                'n_layers': 4,
                'pool': 'max',
                "layers_n_channels": [16, 32, 64, 128],
                'layers_n_non_lins': 2,
                'input_size': (320, 320, 1),
            },
            'val_gen': val_gen_zero,
            'run_id': 'unet_af4_1569210349',
            'reco_function': reco_and_gt_unet_from_val_file,
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
            'val_gen': val_gen_scaled,
            'run_id': 'pdnet_af4_1568384763',
            'reco_function': reco_and_gt_net_from_val_file,
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
            'val_gen': val_gen_scaled,
            'run_id': 'cascadenet_af4_1568926824',
            'reco_function': reco_and_gt_net_from_val_file,
        },
        {
            'name': 'kikinet-sep-16',
            'init_function': full_kiki_net,
            'run_params': {
                'n_convs': 16,
                'n_filters': 48,
                'noiseless': True,
                'activation': lrelu,
            },
            'val_gen': val_gen_scaled,
            'run_id': 'kikinet_sep_I2_af4_1570049560',
            'reco_function': reco_and_gt_net_from_val_file,
            'epoch': 50,
        },
    ]

    def unpack_model(
            init_function=None,
            run_params=None,
            run_id=None,
            epoch=300,
            **dummy_kwargs,
        ):
        model = init_function(**run_params)
        # TODO: better checkpoints getting
        chkpt_path = f'../checkpoints/{run_id}-{epoch}.hdf5'
        model.load_weights(chkpt_path)
        return model

    def metrics_for_params(reco_function=None, val_gen=None, name=None, **net_params):
        model = unpack_model(**net_params)
        metrics = Metrics(METRIC_FUNCS)
        pred_and_gt = [
            reco_function(*val_gen[i], model)
            for i in tqdm_notebook(range(len(val_gen)), desc=f'Val files for {name}')
        ]
        for im_recos, images in tqdm_notebook(pred_and_gt, desc=f'Stats for {name}'):
            metrics.push(images, im_recos)
        return metrics

    def metrics_zfilled():
        metrics = Metrics(METRIC_FUNCS)
        pred_and_gt = [
            reco_and_gt_zfilled_from_val_file(*val_gen_scaled[i])
            for i in tqdm_notebook(range(len(val_gen_scaled)), desc='Val files for z-filled')
        ]
        for im_recos, images in tqdm_notebook(pred_and_gt, desc='Stats for z-filled'):
            metrics.push(images, im_recos)
        return metrics


    metrics = []
    for net_params in all_net_params:
        metrics.append((net_params['name'], metrics_for_params(**net_params)))

    metrics.append(('zfilled', metrics_zfilled()))


    metrics.sort(key=lambda x: x[1].metrics['PSNR'].mean())


    def n_model_params_for_params(
            reco_function=None,
            val_gen=None,
            name=None,
            **net_params,
        ):
        model = unpack_model(**net_params)
        n_params = model.count_params()
        return n_params


    n_params = {}
    for net_params in all_net_params:
        n_params[net_params['name']] = n_model_params_for_params(**net_params)

    n_params['zfilled'] = 0


    def runtime_for_params(reco_function=None, val_gen=None, name=None, **net_params):
        model = unpack_model(**net_params)
        data = val_gen[0]
        start = time.time()
        reco_function(*data, model)
        end = time.time()
        return end - start

    def runtime_zfilled():
        data = val_gen_scaled[0]
        start = time.time()
        reco_and_gt_zfilled_from_val_file(*data)
        end = time.time()
        return end - start

    runtimes = {}
    for net_params in tqdm_notebook(all_net_params):
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

# TODO: code click CLI
