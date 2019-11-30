import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

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
