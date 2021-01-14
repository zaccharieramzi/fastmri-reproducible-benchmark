import tensorflow as tf
from tqdm import tqdm

from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.unet import UnetMultiDomain
from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.mwcnn import MWCNNMultiDomain, DEFAULT_N_FILTERS_PER_SCALE
from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.mwcnn import DEFAULT_N_CONVS_PER_SCALE as default_n_convs_mwcnn


params_per_model = {
    model_name: {} for model_name in 'U-net-multi MWCNN-multi'.split()
}

params_per_model['U-net-multi']['XL'] = dict(
    layers_n_channels=[128, 256, 512, 1024, 1024],
    layers_n_non_lins=2,
)
params_per_model['U-net-multi']['big'] = dict(
    layers_n_channels=[32, 64, 128, 256],
    layers_n_non_lins=2,
)
params_per_model['U-net-multi']['medium'] = dict(
    layers_n_channels=[16, 32, 64, 128],
    layers_n_non_lins=2,
)
params_per_model['U-net-multi']['small'] = dict(
    layers_n_channels=[16, 32, 64],
    layers_n_non_lins=1,
)
params_per_model['U-net-multi']['specs'] = dict(
    model=UnetMultiDomain,
    res=True,
    n_scales='n_layers',
)

params_per_model['MWCNN-multi']['XL'] = dict(
    n_scales=5,
    n_filters_per_scale=[128, 256, 512, 1024, 1024],
    n_convs_per_scale=[2, 2, 2, 2, 2],
    n_first_convs=3,
    first_conv_n_filters=64,
    res=False,
)
params_per_model['MWCNN-multi']['big'] = dict(
    n_scales=3,
    n_filters_per_scale=DEFAULT_N_FILTERS_PER_SCALE,
    n_convs_per_scale=default_n_convs_mwcnn,
    n_first_convs=3,
    first_conv_n_filters=64,
    res=False,
)
params_per_model['MWCNN-multi']['medium'] = dict(
    n_scales=3,
    n_filters_per_scale=[64, 128, 256],
    n_convs_per_scale=default_n_convs_mwcnn,
    n_first_convs=2,
    first_conv_n_filters=32,
    res=False,
)
params_per_model['MWCNN-multi']['small'] = dict(
    n_scales=2,
    n_filters_per_scale=[32, 64],
    n_convs_per_scale=[2, 2],
    n_first_convs=2,
    first_conv_n_filters=32,
    res=False,
)
params_per_model['MWCNN-multi']['specs'] = dict(
    model=MWCNNMultiDomain,
    output_kwarg='n_outputs',
    res=True,
    n_scales='n_scales',
    extra_kwargs=dict(res=False),
)


def get_model_specs(n_primal=1):
    n_outputs = 2*n_primal
    n_inputs = 2*(n_primal+1)
    model_names = sorted(params_per_model.keys())
    for model_name in tqdm(model_names, 'Models'):
        params = params_per_model[model_name]
        model_specs = params['specs']
        model_fun = model_specs['model']
        n_scales_kwarg = model_specs['n_scales']
        res = model_specs['res']
        extra_kwargs = model_specs.get('extra_kwargs', {})
        extra_kwargs.update({'n_outputs': n_outputs})
        model_sizes = sorted(params.keys())
        model_sizes.remove('specs')
        for model_size in tqdm(model_sizes, model_name):
            param_values = params[model_size]
            kwargs = param_values
            kwargs.update(extra_kwargs)
            if n_scales_kwarg == 0:
                n_scales = 0
            else:
                n_scales = kwargs.get(n_scales_kwarg, 4)
                if model_name in 'MWCNN-multi'.split():
                    n_scales += 1
            yield model_name, model_size, model_fun, kwargs, n_inputs, n_scales, res

def build_model_from_specs(model_fun, kwargs, n_inputs):
    model = model_fun(**kwargs)
    model(tf.zeros([1, 32, 32, n_inputs]))
    return model

def get_models(n_primal=1):
    models_specs = get_model_specs(n_primal=n_primal)
    for model_name, model_size, model_fun, kwargs, n_inputs, n_scales, res in models_specs:
        model = build_model_from_specs(model_fun, kwargs, n_inputs)
        yield model_name, model_size, model, n_scales, res
