import os

import pytest

from fastmri_recon.data.utils.tfio import image_and_kspace_from_h5


test_file_single_coil = 'fastmri_recon/tests/fastmri_data/single_coil/file1000002.h5'
test_file_multi_coil = 'fastmri_recon/tests/fastmri_data/multi_coil/file1000001.h5'

@pytest.mark.parametrize('kwargs', [
    {},
    {'inner_slices': 8},
    {'inner_slices': 8, 'rand': True},
])
@pytest.mark.skipif(not os.path.isfile(test_file_single_coil), reason='test single coil file not present for h5 utils.')
def test_image_and_kspace_from_h5(kwargs):
    image_and_kspace_from_h5(**kwargs)(test_file_single_coil)
