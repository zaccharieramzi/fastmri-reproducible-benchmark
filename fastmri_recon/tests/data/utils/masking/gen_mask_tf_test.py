import pytest
import tensorflow as tf

from fastmri_recon.data.utils.masking.gen_mask_tf import gen_mask_tf

@pytest.mark.parametrize('fixed_masks', [True, False])
@pytest.mark.parametrize('multicoil', [True, False])
def test_gen_mask_tf(fixed_masks, multicoil):
    kspace = tf.random.uniform([2, 5, 64, 32])
    kspace = tf.cast(kspace, tf.complex64)
    accel_factor = 2
    mask = gen_mask_tf(kspace, accel_factor, multicoil, fixed_masks)
    if fixed_masks:
        mask_again = gen_mask_tf(kspace, accel_factor, multicoil, fixed_masks)
        tf_tester = tf.test.TestCase()
        tf_tester.assertEqual(mask, mask_again)
