from tensorflow.python.client import device_lib


def gpu_index_from_submodel_index(n_gpus, n_submodels, submodel_i):
    if n_submodels <= n_gpus:
        return submodel_i
    else:
        n_left_over_models = n_submodels % n_gpus
        n_submodels_per_gpu = n_submodels // n_gpus
        # for i_gpu < n_left_over_models, there is n_submodels_per_gpu + 1
        # submodels per GPU, otherwise n_submodels_per_gpu
        i_gpu = submodel_i // (n_submodels_per_gpu + 1)
        if i_gpu >= n_left_over_models:
            i_gpu = submodel_i // n_submodels_per_gpu
        return i_gpu

def get_gpus():
    # from https://gist.github.com/jovianlin/b5b2c4be45437992df58a7dd13dbafa7
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
