import pytest

from fastmri_recon.models.utils.gpu_placement import gpu_index_from_submodel_index

@pytest.mark.parametrize('n_gpus, n_submodels, submodel_i, expected_i_gpu',[
    (1, 1, 0, 0),
    (8, 2, 2, 2),
    (4, 8, 6, 3),
    (4, 9, 2, 0),
    (4, 9, 3, 1),
    (3, 10, 9, 2),
])
def test_gpu_index_from_submodel_index(n_gpus, n_submodels, submodel_i, expected_i_gpu):
    i_gpu_res = gpu_index_from_submodel_index(n_gpus, n_submodels, submodel_i)
    assert expected_i_gpu == i_gpu_res
