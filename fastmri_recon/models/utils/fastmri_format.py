import tensorflow as tf
from tensorflow.keras.layers import Lambda

def _tf_crop(im, cropx=320, cropy=None):
    if cropy is None:
        cropy = cropx
    im_shape = tf.shape(im)
    y = im_shape[1]
    x = im_shape[2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    starty = tf.cond(
        tf.equal(tf.math.mod(cropy, 2), 0),
        lambda: starty,
        lambda: starty -1,
    )
    im = im[:, starty:starty+cropy, startx:startx+cropx, :]
    return im

def tf_fastmri_format(image):
    image = Lambda(lambda x: _tf_crop(tf.math.abs(x)), name='cropping', output_shape=(320, 320, 1))(image)
    return image

def general_fastmri_format(image, output_shape=None):
    abs_image = tf.math.abs(image)
    if output_shape is None:
        cropy = cropx = 320
    else:
        cropx = output_shape[-2]
        cropy = output_shape[-1]
    cropped_image = _tf_crop(abs_image, cropx=cropx, cropy=cropy)
    return cropped_image
