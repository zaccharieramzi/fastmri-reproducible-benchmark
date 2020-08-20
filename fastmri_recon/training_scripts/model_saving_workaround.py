from tensorflow.keras.callbacks import ModelCheckpoint


class ModelCheckpointWorkAround(ModelCheckpoint):
    def set_model(self, model):
        # Work around, so that the if at
        # https://github.com/tensorflow/tensorflow/blob/1186e3f2098793952aa82bf356dfe51b967fb26c/tensorflow/python/keras/callbacks.py#L1189
        # is skipped, so that self.save_weights_only remains False.
        self.model = model
