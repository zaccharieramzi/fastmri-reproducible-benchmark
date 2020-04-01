import os

import pytest

from fastmri_recon.data.utils.h5 import (
    from_train_file_to_image_and_kspace,
    from_train_file_to_image_and_kspace_and_contrast,
    from_file_to_kspace,
    from_file_to_contrast,
)

test_file_single_coil = 'file1000002.h5'

@pytest.mark.skipif(not os.path.isfile(test_file_single_coil), reason='test file not present for h5 utils.')
def test_all_functions_single_coil():
    functions_sliceable = [
        from_train_file_to_image_and_kspace,
        from_train_file_to_image_and_kspace_and_contrast,
        from_file_to_kspace,
    ]

    functions_to_test = functions_sliceable + [from_file_to_contrast]
    for fun in functions_to_test:
        fun(test_file_single_coil)
    for fun in functions_sliceable:
        # this gets the 10-th slice of the volume
        fun(test_file_single_coil, slices=(10))
