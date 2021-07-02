#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd

plt.style.use('science')
plt.rcParams['figure.figsize'] = (5.5, 2.8)
plt.rcParams['font.size'] = 8


# In[2]:


CONTRASTS = ['CORPD_FBK', 'CORPDFS_FBK']
ACQ_TYPES = ['radial', 'spiral']
def get_contrast(filename):
    for contrast in CONTRASTS:
        if contrast in filename:
            return contrast
        
def get_acq_type(filename):
    for acq_type in ACQ_TYPES:
        if acq_type in filename:
            return acq_type
        
def get_af(filename):
    if 'af8' in filename:
        return 8
    else:
        return 4
        
def get_model_name(filename):
    if 'ncpdnet' in filename:
        if is_dcomp(filename):
            return 'ncpdnet-dcomp'
        else:
            return 'ncpdnet'
    elif 'unet' in filename:
        return 'unet'
    elif 'pdnet' in filename:
        return 'pdnet'
    elif 'None' in filename:
        return 'adj-dc'
        
def is_dcomp(filename):
    return 'dcomp' in filename

res_map = {}
for contrast in CONTRASTS:
    res_map[contrast] = {acq_type: {
        af: {} for af in [4, 8]
    } for acq_type in ACQ_TYPES}
    
res_path = Path('res_nc/')
res_filenames = sorted(list(res_path.glob('*.csv')))

for filename in res_filenames:
    contrast = get_contrast(filename.stem)
    acq_type = get_acq_type(filename.stem)
    model_name = get_model_name(filename.stem)
    af = get_af(filename.stem)
    res_map[contrast][acq_type][af][model_name] = filename


# In[3]:


for af in [4, 8]:
    fig = plt.figure()
    keys = ['PSNR', 'SSIM']
    gs = gridspec.GridSpec(2, 2, wspace=0.2, hspace=.05, height_ratios=[0.8, 0.2])
    inner_gses = [
        gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, i_key], hspace=.05, wspace=0.05)
        for i_key in range(len(keys))
    ]
    model_order = ['adj-dc', 'pdnet', 'ncpdnet', 'unet', 'ncpdnet-dcomp']
    contrast_map = {
        'CORPD_FBK': 'PD',
        'CORPDFS_FBK': 'PDFS',
    }
    acq_type_map = {
        'radial': 'Radial',
        'spiral': 'Spiral',
    }
    model_name_map = {
        'adj-dc': 'Adjoint + DComp',
        'pdnet': 'PDNet-Gridded',
        'ncpdnet': 'NCPDNet',
        'unet': 'U-Net',
        'ncpdnet-dcomp': 'NCPDNet + DComp',
    }
    colors = [f'C{i}' for i in range(len(model_order))]
    reference_ax_for_key = {}
    for i_key, key in enumerate(keys):
        inner_gs = inner_gses[i_key]
        title_ax = fig.add_subplot(inner_gs[:, :])
        title_ax.set_title(key, y=1.05)
        title_ax.axis('off')
        for i_contrast, contrast in enumerate(CONTRASTS):
            for i_acq, acq_type in enumerate(ACQ_TYPES):
                handles = []
                grid_spec_loc = inner_gs[i_contrast, i_acq]
                if i_contrast == 0 and i_acq == 0:
                    ax = fig.add_subplot(grid_spec_loc)
                    reference_ax_for_key[key] = ax
                else:
                    ref_ax = reference_ax_for_key[key]
                    ax = fig.add_subplot(
                        grid_spec_loc, 
                        sharey=ref_ax,
                    )
                    if i_acq > 0:
                        plt.setp(ax.get_yticklabels(), visible=False)
                results = res_map[contrast][acq_type][af]
                if not results:
                    continue
                labels = []
                for i_model, (model_name, color) in enumerate(zip(model_order, colors)):
                    try:
                        df = pd.read_csv(results[model_name], index_col=0)
                    except KeyError:
                        continue
                    data = df[key]
                    label = model_name_map[model_name]
                    bplot = ax.boxplot(
                        [data], 
                        labels=[label], 
                        patch_artist=True,
                        positions=[i_model],
                        boxprops=dict(facecolor=color),
                        medianprops=dict(color='red'),
                        widths=0.5,
                    )
                    labels.append(label)
                    handles.append(bplot['boxes'][0])
                ax.set_xticklabels(labels, rotation=75)
                if i_acq == 0 and i_key == 0:
                    ax.set_ylabel(contrast_map[contrast])
                if i_contrast == 0:
                    ax.set_title(acq_type_map[acq_type])
                ax.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
    # gs = axs[2, 0].get_gridspec()
    # # remove the underlying axes
    # for ax in axs[-1, :]:
    #     ax.remove()
    axlegend = fig.add_subplot(gs[-1, :])
    axlegend.axis('off')
    axlegend.legend(handles, labels, loc='center', ncol=3, handlelength=1.5, handletextpad=.2)
    figname = 'single_coil'
    if af == 8:
        figname += '_af8'
    figname += '.pdf'
    fig.savefig(figname, dpi=300);


# In[ ]:




