import pytest

from fastmri_recon.models.subclassed_models.dip import DIPBase

@pytest.mark.parametrize('bn', [True, False])
def test_init(bn):
    DIPBase(bn=bn)
