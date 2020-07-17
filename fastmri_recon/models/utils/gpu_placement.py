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
