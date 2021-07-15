from pathlib import Path
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (5.5, 2.7)
plt.rcParams['font.size'] = 8
plt.style.use(['science', 'ieee'])


## Data extraction

fig_dir = Path('figures')
fig_subdirs = [d for d in fig_dir.iterdir() if d.is_dir()]

model_names = ['adj-dcomp', 'dip', 'pdnet-gridded', 'pdnet-norm', 'unet', 'pdnet-dcomp']
trajectories = ['radial', 'spiral']

def is_sc(dirname):
    return 'sc' in dirname

def is_mc(dirname):
    return 'mc' in dirname

def is_dip(dirname):
    return 'dip' in dirname

def is_brain(dirname):
    return 'brain' in dirname

def is_rev(dirname):
    return 'rev' in dirname

def is_zoom(name):
    return 'zoom' in name

def get_af(name):
    if 'af8' in name:
        return 8
    else:
        return 4
    
def is_error(filename):
    return 'residu' in filename
    
def get_traj(filename):
    for traj in trajectories:
        if traj in filename:
            return traj
    
def get_model_name(filename):
    for model_name in model_names + ['gt']:
        if model_name in filename:
            return model_name
        
# fig res maps
# res_map[is_brain][rev][is_sc][is_zoom][af][traj][model_name][error]
res_map = {}
tr_fa = (True, False)
for brain in tr_fa:
    res_map[brain] = {}
    for rev in tr_fa:
        res_map[brain][rev] = {}
        for sc in tr_fa:
            res_map[brain][rev][sc] = {}
            for zoom in tr_fa:
                res_map[brain][rev][sc][zoom] = {}
                for af in [4, 8]:
                    res_map[brain][rev][sc][zoom][af] = {}
                    for traj in trajectories:
                        res_map[brain][rev][sc][zoom][af][traj] = {}
                        for model_name in model_names + ['gt']:
                            res_map[brain][rev][sc][zoom][af][traj][model_name] = {}
                        
excluded_dir = ['grappa', 'xpdnet']

for subdir in fig_subdirs:
    subname = subdir.stem
    if any(ed in subname for ed in excluded_dir):
        continue
    brain = is_brain(subname)
    dip = is_dip(subname)
    sc = is_sc(subname)
    rev = is_rev(subname)
    zoom = is_zoom(subname)
    af = get_af(subname)
    for figpath in subdir.glob('*.png'):
        figname = figpath.stem
        if dip:
            af = get_af(figname)
        traj = get_traj(figname)
        model_name = get_model_name(figname)
        error = is_error(figname)
        try:
            res_map[brain][rev][sc][zoom][af][traj][model_name][error] = figpath
        except KeyError:
            pass
        if model_name in ['dip', 'adj-dcomp']:
            res_map[brain][not rev][sc][zoom][af][traj][model_name][error] = figpath
        if model_name == 'gt':
            for new_traj in trajectories:
                for new_rev in tr_fa:
                    for new_af in [4, 8]:
                        res_map[brain][new_rev][sc][zoom][new_af][new_traj][model_name][error] = figpath

model_name_map = {
    'adj-dcomp': r'\textbf{Adj. + DCp}',
    'dip': r'\textbf{DIP}',
    'pdnet-gridded': r'\textbf{grid. PDNet}',
    'pdnet-norm': r'\textbf{PDNet}',
    'pdnet-dcomp': r'\textbf{\emph{NCPDNet}}',
    'unet': r'\textbf{U-net}',
    'gt': r'\textbf{Reference}',
}

margins = slice(50, -50)
figwidth = 9
for brain in tr_fa:
    for rev in tr_fa:
        if brain and rev:
            continue
        for sc in tr_fa:
            if (sc and rev) or (sc and brain):
                continue
            for zoom in tr_fa:
                for af in [4, 8]:
                    if af == 8 and (rev or brain):
                        continue
                    for traj in trajectories:
                        res = res_map[brain][rev][sc][zoom][af][traj]
                        models = [k for k, v in res.items() if v]
                        fig, axs = plt.subplots(2, len(models), figsize=(figwidth, figwidth / len(models) * 2), gridspec_kw={'hspace': 0.001, 'wspace': 0.001})
                        model_counter = 0
                        for model in ['gt'] + model_names:
                            if model not in models:
                                continue
                            recon = plt.imread(res[model][False])
                            axs[0, model_counter].imshow(recon[margins, margins], aspect='auto')
                            axs[0, model_counter].set_title(model_name_map[model], fontsize=11, pad=0.01)
                            if model != 'gt':
                                error = plt.imread(res[model][True])
                                axs[1, model_counter].imshow(error[margins, margins], aspect='auto')
                            for ax in axs[:, model_counter]:
                                ax.axis('off')
                            model_counter += 1
                        fig_name = 'quali'
                        if brain:
                            fig_name += '_brain'
                        if rev:
                            fig_name += '_rev'
                        if not sc:
                            fig_name += 'mc'
                        if zoom:
                            fig_name += '_zoom'
                        fig_name += f'_af{af}_{traj}.pdf'
                        plt.savefig(fig_dir / fig_name, dpi=300)
