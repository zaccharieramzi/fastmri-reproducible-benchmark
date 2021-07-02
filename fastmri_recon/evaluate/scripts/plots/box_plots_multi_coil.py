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


CONTRASTS_KNEE = ['CORPD_FBK', 'CORPDFS_FBK']
CONTRASTS_BRAIN = ['AXT2', 'AXT1POST', 'AXT1PRE', 'AXFLAIR', 'AXT1']
ACQ_TYPES = ['radial', 'spiral']
def get_contrast(filename):
    for contrast in CONTRASTS_KNEE + CONTRASTS_BRAIN:
        if contrast in filename:
            return contrast
        
def get_eval_acq_type(filename):
    for acq_type in ACQ_TYPES:
        if ('eval_on_' + acq_type) in filename:
            return acq_type
        
def get_train_acq_type(filename):
    run_id = filename.split('eval_on')[0]
    for acq_type in ACQ_TYPES:
        if acq_type in run_id:
            return acq_type
    return get_eval_acq_type(filename)
        
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
    elif 'dip' in filename:
        return 'dip'
    elif 'None' in filename:
        return 'adj-dc'
    
def get_af(filename):
    if '_af8' in filename:
        return 8
    else:
        return 4
        
def is_dcomp(filename):
    return 'dcomp' in filename

def is_brain(filename):
    return 'brain' in filename

res_map = {}
for contrast in CONTRASTS_KNEE + CONTRASTS_BRAIN:
    res_map[contrast] = {acq_type: {
        brain: {
            reverse: {
                af: {} for af in [4, 8]
            } for reverse in [True, False]
        }
        for brain in [True, False]
    } for acq_type in ACQ_TYPES}
    
res_path = Path('res_ncmc/')
res_filenames = sorted(list(res_path.glob('*.csv')))

for filename in res_filenames:
    contrast = get_contrast(filename.stem)
    brain = is_brain(filename.stem)
    train_acq_type = get_train_acq_type(filename.stem)
    eval_acq_type = get_eval_acq_type(filename.stem)
    af = get_af(filename.stem)
    reverse = train_acq_type != eval_acq_type
    model_name = get_model_name(filename.stem)
    res_map[contrast][eval_acq_type][brain][reverse][af][model_name] = filename
    if model_name == 'adj-dc' or model_name == 'dip':
        res_map[contrast][eval_acq_type][brain][not reverse][af][model_name] = filename


# In[3]:


res_map


# In[8]:


model_order = ['adj-dc', 'dip', 'unet', 'ncpdnet-dcomp']
contrast_map = {
    'CORPD_FBK': 'PD',
    'CORPDFS_FBK': 'PDFS',
    'AXT1': 'T1',
    'AXT1PRE': 'T1 - PRE',
    'AXT1POST': 'T1 - POST',
    'AXT2': 'T2',
    'AXFLAIR': 'FLAIR',
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
    'dip': 'DIP',
}
colors = 'C0 C5 C3 C4'.split()
for (brain, reverse, af) in [
    (True, False, 4), 
    (False, True, 4), 
    (False, False, 4),
    (False, False, 8),
]:
    if brain:
        fig = plt.figure(figsize=(5.5, (5/2) * 2.8))
        ratio = [(5/2)*0.8, 0.2]
    else:
        fig = plt.figure()
        ratio = [0.8, 0.2]
    keys = ['PSNR', 'SSIM']
    contrasts = CONTRASTS_BRAIN if brain else CONTRASTS_KNEE
    gs = gridspec.GridSpec(2, 2, wspace=0.2, hspace=.05, height_ratios=ratio)
    inner_gses = [
        gridspec.GridSpecFromSubplotSpec(len(contrasts), 2, subplot_spec=gs[0, i_key], hspace=.05, wspace=0.05)
        for i_key in range(len(keys))
    ]    
    reference_ax_for_key = {}
    for i_key, key in enumerate(keys):
        inner_gs = inner_gses[i_key]
        title_ax = fig.add_subplot(inner_gs[:, :])
        title_ax.set_title(key, y=1.05)
        title_ax.axis('off')
        labels = []
        handles = []
        for i_contrast, contrast in enumerate(contrasts):
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
                results = res_map[contrast][acq_type][brain][reverse][af]
                if not results:
                    continue
                labels = []
                model_counter = 0
                for model_name, color in zip(model_order, colors):
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
                        positions=[model_counter],
                        boxprops=dict(facecolor=color),
                        medianprops=dict(color='red'),
                        widths=0.5,
                    )
                    model_counter += 1
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
    axlegend.legend(handles, labels, loc='center', ncol=len(model_order), handlelength=1.5, handletextpad=.2)
    # fig.suptitle(key, y=1.0)
    fig_name = 'multicoil'
    if brain:
        fig_name += '_brain'
    if reverse:
        fig_name += '_reverse'
    if af == 8:
        fig_name += '_af8'
    fig_name += '.pdf'
    fig.savefig(fig_name, dpi=300);


# In[ ]:




