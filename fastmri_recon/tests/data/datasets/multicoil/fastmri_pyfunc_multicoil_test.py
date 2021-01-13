try:
    import ismrmrd
except ImportError:
    ismrmrd_not_avail = True
else:
    ismrmrd_not_avail = False
import numpy as np
import pytest
import tensorflow as tf

# NOTE: functions beginning with test will be marked as tests by pytest
# so I need to import them with a slightly different name
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import test_masked_kspace_dataset_from_indexable as _test_masked_kspace_dataset_from_indexable
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import test_filenames as _test_filenames
from fastmri_recon.data.utils.h5 import from_test_file_to_mask_and_contrast

kspace_shape = [2, 15, 640, 322, 1]
file_contrast = 'CORPD_FBK'

@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape', [
    ({}, kspace_shape[0:1] + kspace_shape[2:]),
    ({'mask_type': 'equidistant'}, kspace_shape[0:1] + kspace_shape[2:]),
    ({'parallel': False}, kspace_shape),
    ({'inner_slices': 1}, [1,] + kspace_shape[2:]),
    ({'inner_slices': 1, 'rand': True}, [1,] + kspace_shape[2:]),
    ({'inner_slices': 1, 'rand': True, 'parallel': False}, [1,] + kspace_shape[1:]),
    ({'contrast': file_contrast, 'parallel': False}, kspace_shape),
    ({'n_samples': 1, 'parallel': False}, kspace_shape),
])
def test_train_masked_kspace_dataset_from_indexable(create_full_fastmri_test_tmp_dataset, ds_kwargs, expected_kspace_shape):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_train']
    ds = train_masked_kspace_dataset_from_indexable(path, AF=1, **ds_kwargs)
    if ds_kwargs.get('parallel', True):
        (kspace, mask), image = next(iter(ds))
    else:
        (kspace, mask, smaps), image = next(iter(ds))
    # shape verifications
    assert kspace.shape.as_list() == expected_kspace_shape
    assert mask.shape.as_list() == [expected_kspace_shape[0]] + [1 for _ in expected_kspace_shape[1:-2]] + [expected_kspace_shape[-2]]
    if ds_kwargs.get('parallel', True):
        assert image.shape.as_list() == expected_kspace_shape
    else:
        assert smaps.shape.as_list() == expected_kspace_shape[:-1]
        assert image.shape.as_list() == expected_kspace_shape[0:1] + [320, 320, 1]
    # content verifications
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllInSet(mask, [1])  # this because we set af to 1
    # TODO: implement adjoint fourier multicoil

@pytest.mark.skipif(ismrmrd_not_avail, reason='ismrmrd not installed')
@pytest.mark.parametrize('ds_kwargs, expected_kspace_shape', [
    ({}, kspace_shape),
    ({'contrast': file_contrast}, kspace_shape),
    ({'n_samples': 1}, kspace_shape),
])
def test_test_masked_kspace_dataset_from_indexable(create_full_fastmri_test_tmp_dataset, ds_kwargs, expected_kspace_shape):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_test']
    af = create_full_fastmri_test_tmp_dataset['af_multi_coil'][0]
    if af > 5.5:
        AF = 8
    else:
        AF = 4
    ds = _test_masked_kspace_dataset_from_indexable(path, AF=AF, **ds_kwargs)
    kspace, mask, smaps = next(iter(ds))
    # shape verifications
    assert kspace.shape.as_list() == expected_kspace_shape
    assert mask.shape.as_list() == [expected_kspace_shape[0]] + [1 for _ in expected_kspace_shape[1:-2]] + [expected_kspace_shape[-2]]
    assert smaps.shape.as_list() == expected_kspace_shape[:-1]
    # content verifications
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllInSet(mask, [1, 0])  # this because we set af to 1

@pytest.mark.skipif(ismrmrd_not_avail, reason='ismrmrd not installed')
@pytest.mark.parametrize('ds_kwargs', [
    {},
    {'n_samples': 1},
])
def test_test_filenames(create_full_fastmri_test_tmp_dataset, ds_kwargs):
    path = create_full_fastmri_test_tmp_dataset['fastmri_tmp_multicoil_test']
    af = create_full_fastmri_test_tmp_dataset['af_multi_coil'][0]
    if af > 5.5:
        AF = 8
    else:
        AF = 4
    files_ds = _test_filenames(path, AF=AF, **ds_kwargs)
    ds = _test_masked_kspace_dataset_from_indexable(path, AF=AF, **ds_kwargs)
    filename = next(iter(files_ds))
    _, mask, _ = next(iter(ds))
    mask_from_file, _ = from_test_file_to_mask_and_contrast(filename.numpy())
    np.testing.assert_equal(np.squeeze(mask).astype(bool)[0], mask_from_file)
