import os

import pytest
import tensorflow as tf

from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.models.utils.fastmri_format import tf_fastmri_format
from fastmri_recon.models.utils.fourier import tf_unmasked_adj_op

test_file_single_coil = 'fastmri_recon/tests/fastmri_data/single_coil/file1000002.h5'
kspace_shape = [38, 640, 368, 1]
file_contrast = 'CORPDFS_FBK'

@pytest.mark.skipif(not os.path.isfile(test_file_single_coil), reason='test single coil file not present for single dataset.')
@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape', [
    ({}, kspace_shape),
    ({'inner_slices': 8}, [8,] + kspace_shape[1:]),
    ({'inner_slices': 8, 'rand': True}, [1,] + kspace_shape[1:]),
    ({'contrast': file_contrast}, kspace_shape),
    ({'n_samples': 1}, kspace_shape)
])
def test_train_masked_kspace_dataset_from_indexable(ds_kwargs, expected_kspace_shape):
    ds = train_masked_kspace_dataset_from_indexable('fastmri_recon/tests/fastmri_data/single_coil/', AF=1, **ds_kwargs)
    (kspace, mask), image = next(iter(ds))
    reconstructed_image = tf_fastmri_format(tf_unmasked_adj_op(kspace))
    # shape verifications
    assert kspace.shape.as_list() == expected_kspace_shape
    assert mask.shape.as_list() == [1 for _ in expected_kspace_shape[:-2]] + [expected_kspace_shape[-2]]
    assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]
    # content verifications
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllInSet(mask, [1 + 0.j])  # this because we set af to 1
    tf_tester.assertAllClose(image, reconstructed_image)
