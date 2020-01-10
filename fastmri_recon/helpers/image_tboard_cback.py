"""Inspired by https://github.com/sicara/tf-explain/blob/master/tf_explain/callbacks/grad_cam.py"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class TensorBoardImage(Callback):
    def __init__(self, log_dir, image, model_input):
        super().__init__()
        self.log_dir = log_dir
        self.image = image
        self.model_input = model_input

    def set_model(self, model):
        self.model = model
        self.writer = tf.summary.create_file_writer(self.log_dir, filename_suffix='images')

    def on_train_begin(self, _):
        self.write_image(self.image, 'Original Image', 0)

    def on_train_end(self, _):
        self.writer.close()

    def write_image(self, image, tag, epoch):
        image_to_write = np.copy(image)
        image_to_write -= image_to_write.min()
        image_to_write /= image_to_write.max()
        with self.writer.as_default():
            tf.summary.image(tag, image_to_write, step=epoch)

    def on_epoch_end(self, epoch, logs={}):
        reconstructed_image = self.model.predict_on_batch(self.model_input)
        if isinstance(reconstructed_image, list):
            reconstructed_image = reconstructed_image[0]
        self.write_image(reconstructed_image, 'Reconstructed Image', epoch)
