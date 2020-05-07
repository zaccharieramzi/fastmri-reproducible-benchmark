import pytest
import tensorflow as tf

from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
# from fastmri_recon.models.utils.fastmri_format import tf_fastmri_format
# from fastmri_recon.models.utils.fourier import tf_unmasked_adj_op


kspace_shape = [2, 640, 322, 1]
file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape', [
    ({}, kspace_shape),
    ({'inner_slices': 1}, [1,] + kspace_shape[1:]),
    ({'inner_slices': 1, 'rand': True}, [1,] + kspace_shape[1:]),
    ({'contrast': file_contrast}, kspace_shape),
    ({'n_samples': 1}, kspace_shape)
])
def test_train_masked_kspace_dataset_from_indexable(create_full_fastmri_test_tmp_dataset, ds_kwargs, expected_kspace_shape):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_singlecoil_train']
    ds = train_masked_kspace_dataset_from_indexable(path, AF=1, **ds_kwargs)
    (kspace, mask), image = next(iter(ds))
    # shape verifications
    assert kspace.shape.as_list() == expected_kspace_shape
    assert mask.shape.as_list() == [expected_kspace_shape[0]] + [1 for _ in expected_kspace_shape[1:-2]] + [expected_kspace_shape[-2]]
    assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]
    # content verifications
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllInSet(mask, [1 + 0.j])  # this because we set af to 1
    # NOTE: for now let's put this aside as we don't have the fastMRI data in repo
    # tf_tester.assertAllClose(image, reconstructed_image)
