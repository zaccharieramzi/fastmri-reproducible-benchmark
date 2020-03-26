import tensorflow.keras.backend as K


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)
