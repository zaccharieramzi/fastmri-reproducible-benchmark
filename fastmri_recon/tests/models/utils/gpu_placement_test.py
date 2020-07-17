import pytest

from fastmri_recon.models.utils.gpu_placement import gpu_index_from_submodel_index

@pytest.mark.parametrize
def test_gpu_index_from_submodel_index():
    i_gpu_res = gpu_index_from_submodel_index(4, 9, )
