import tensorflow as tf
from tensorflow.keras.layers import Lambda

def _tf_crop(im, crop=320):
    im_shape = tf.shape(im)
    y = im_shape[1]
    x = im_shape[2]
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    im = im[:, starty:starty+crop, startx:startx+crop, :]
    return im

def tf_fastmri_format(image):
    image = Lambda(lambda x: _tf_crop(tf.math.abs(x)), name='cropping', output_shape=(320, 320, 1))(image)
    return image
