"""Inspired by https://stackoverflow.com/a/49363251/4332585"""
import io

from keras.callbacks import Callback
import numpy as np
from PIL import Image
from skimage.util import img_as_ubyte
import tensorflow as tf

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    _, height, width, channel = tensor.shape
    tensor = tensor[0]
    tensor_normalized = tensor - tensor.min()
    tensor_normalized /= tensor_normalized.max()
    tensor_normalized = img_as_ubyte(tensor_normalized)
    tensor_squeezed = np.squeeze(tensor_normalized)
    image = Image.fromarray(tensor_squeezed)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    summary = tf.Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string,
    )
    return summary

class TensorBoardImage(Callback):
    def __init__(self, log_dir, image, model_input):
        super().__init__()
        self.log_dir = log_dir
        self.image = image
        self.model_input = model_input

    def set_model(self, model):
        self.model = model
        self.writer = tf.summary.FileWriter(self.log_dir, filename_suffix='images')

    def on_train_begin(self, _):
        self.write_image(self.image, 'Original Image', 0)

    def on_train_end(self, _):
        self.writer.close()

    def write_image(self, image, tag, epoch):
        image = make_image(image)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=image)])
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs={}):
        [reconstructed_image, _] = self.model.predict_on_batch(self.model_input)
        self.write_image(reconstructed_image, 'Reconstructed Image', epoch)
