import tensorflow as tf


def crop_center(img, cropx, cropy=None):
    # taken from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image/39382475
    if cropy is None:
        cropy = cropx
    y, x = img.shape[-2:]
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    return img[..., starty:starty+cropy, startx:startx+cropx]

def adjust_image_size(image, target_image_size, multicoil=False):
    height = tf.shape(image)[-2]
    width = tf.shape(image)[-1]
    n_slices = tf.shape(image)[0]
    reshaped_image = tf.reshape(image, [-1, height, width])
    target_height = target_image_size[0]
    target_width = target_image_size[1]
    padded_image = tf.image.resize_with_crop_or_pad(
        reshaped_image,
        target_height,
        target_width,
    )
    if multicoil:
        final_shape = [n_slices, target_height, target_width]
    else:
        final_shape = [n_slices, -1, target_height, target_width]
    reshaped_padded_image = tf.reshape(padded_image, final_shape)
    return reshaped_padded_image
