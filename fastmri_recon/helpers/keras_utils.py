import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from .utils import keras_psnr, keras_ssim


def default_model_compile(model, lr):
    model.compile(
        optimizer=Adam(lr=lr, clipnorm=1.),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
    )

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def mean_output(y_true, y_pred):
    return K.mean(y_pred)

def discriminator_accuracy(y_true, y_pred):
    y_pred_binary = K.cast(K.less(y_pred, 0.5), 'float32')
    y_true_binary = K.cast(K.greater(y_true, 0), 'float32')
    return K.mean(K.cast(K.equal(y_true_binary, y_pred_binary), 'float32'))

# functions from https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
# not ported to tf 2.0 (or at least not found easily)
def iter_sequence_infinite(seq):
    """Iterate indefinitely over a Sequence.
    # Arguments
        seq: Sequence object
    # Returns
        Generator yielding batches.
    """
    while True:
        for item in seq:
            yield item


def is_sequence(seq):
    """Determine if an object follows the Sequence API.
    # Arguments
        seq: a possible Sequence object
    # Returns
        boolean, whether the object follows the Sequence API.
    """
    # TODO Dref360: Decide which pattern to follow. First needs a new TF Version.
    return (getattr(seq, 'use_sequence_api', False)
            or set(dir(Sequence())).issubset(set(dir(seq) + ['use_sequence_api'])))

# from https://github.com/keras-team/keras/blob/master/keras/utils/metrics_utils.py
def to_list(x):
    if isinstance(x, list):
        return x
    return [x]
