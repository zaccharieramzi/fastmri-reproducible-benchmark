import os
import pickle as pkl

from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K


# taken from https://github.com/tensorflow/tensorflow/issues/39679
class ModelCheckpointWorkAround(ModelCheckpoint):
    def __init__(self, filepath, save_optimizer=True, **kwargs):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.save_optimizer = save_optimizer
        super(ModelCheckpointWorkAround, self).__init__(filepath=filepath, **kwargs)

    def set_model(self, model):
        # Work around, so that the if at
        # https://github.com/tensorflow/tensorflow/blob/1186e3f2098793952aa82bf356dfe51b967fb26c/tensorflow/python/keras/callbacks.py#L1189
        # is skipped, so that self.save_weights_only remains False.
        self.model = model

    def _save_model(self, epoch, batch, logs):
        # Save the model with super
        super(ModelCheckpointWorkAround, self)._save_model(epoch, batch, logs)
        if self.save_optimizer:
            # Save the optimizer
            folder = os.path.dirname(self._get_file_path(epoch, batch, logs))
            symbolic_weights = getattr(self.model.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            with open(os.path.join(folder, 'optimizer.pkl'), 'wb') as f:
                pkl.dump(weight_values, f)
