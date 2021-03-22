import tensorflow as tf

from fastmri_recon.data.utils.crop import adjust_image_size


def test_adjust_image_size():
    orig_images = [[
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ]
    ]]
    orig_images = tf.constant(orig_images)
    adjusted_images = adjust_image_size(orig_images, (4, 4), multicoil=True)
    assert adjusted_images.shape == (1, 2, 4, 4)
    expected_images = [[
        [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 5, 6, 0],
            [0, 7, 8, 0],
            [0, 0, 0, 0],
        ]
    ]]
    expected_images = tf.constant(expected_images)
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllEqual(adjusted_images, expected_images)
