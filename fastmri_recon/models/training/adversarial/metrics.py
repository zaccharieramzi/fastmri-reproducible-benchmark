import tensorflow.keras.backend as K

def mean_output(y_true, y_pred):
    return K.mean(y_pred)

def discriminator_accuracy(y_true, y_pred):
    y_pred_binary = K.cast(K.less(y_pred, 0.5), 'float32')
    y_true_binary = K.cast(K.greater(y_true, 0), 'float32')
    return K.mean(K.cast(K.equal(y_true_binary, y_pred_binary), 'float32'))
