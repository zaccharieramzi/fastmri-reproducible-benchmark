import tensorflow as tf


def gpu_index_from_submodel_index(n_gpus, n_submodels, submodel_i):
    if n_submodels <= n_gpus:
        return submodel_i
    else:
        n_left_over_models = n_submodels % n_gpus
        n_submodels_per_gpu = n_submodels // n_gpus
        i_limit_overflow = n_left_over_models * (n_submodels_per_gpu + 1)
        if submodel_i < i_limit_overflow:
            i_gpu = submodel_i // (n_submodels_per_gpu + 1)
        else:
            i_gpu = (submodel_i - n_left_over_models) // n_submodels_per_gpu
        return i_gpu

def get_gpus():
    local_device_protos = tf.config.list_logical_devices('GPU')
    return [x.name for x in local_device_protos]
