import pytest
import tensorflow as tf

from fastmri_recon.data.datasets.fastmri_pyfunc_denoising import train_noisy_dataset_from_indexable


file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('ds_kwargs', [
    {},
    {'inner_slices': 1},
    {'inner_slices': 1, 'rand': True},
    {'contrast': file_contrast},
    {'n_samples': 1},
])
def test_train_masked_kspace_dataset_from_indexable(create_full_fastmri_test_tmp_dataset, ds_kwargs):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = train_noisy_dataset_from_indexable(path, noise_std=0, **ds_kwargs)
    image_noisy, image = next(iter(ds))
    # content verifications
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllEqual(image_noisy, image)
