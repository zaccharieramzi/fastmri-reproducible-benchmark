import pytest

from fastmri_recon.data.utils.tfio import image_and_kspace_from_h5


@pytest.mark.parametrize('kwargs', [
    {},
    {'inner_slices': 1},
    {'inner_slices': 1, 'rand': True},
])
def test_image_and_kspace_from_h5(create_full_fastmri_test_tmp_dataset, kwargs):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    image_and_kspace_from_h5(**kwargs)(path + 'train_singlecoil_0.h5')
