from fastmri_recon.data.utils.h5 import (
    from_train_file_to_image_and_kspace,
    from_train_file_to_image_and_kspace_and_contrast,
    from_file_to_kspace,
    from_file_to_contrast,
)

test_file = 'file1000002.h5'

def test_all_functions():
    functions_to_test = [
        from_train_file_to_image_and_kspace,
        from_train_file_to_image_and_kspace_and_contrast,
        from_file_to_kspace,
        from_file_to_contrast,
    ]
    for fun in functions_to_test:
        fun(test_file)
