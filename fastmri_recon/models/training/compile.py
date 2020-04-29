import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import Adam

from ...evaluate.metrics.tf_metrics import keras_psnr, keras_ssim


def default_model_compile(model, lr, loss='mean_absolute_error'):
    opt_kwargs = {}
    precision_policy = mixed_precision.global_policy()
    if precision_policy.loss_scale is None:
        opt_kwargs['clipnorm'] = 1.
    if loss == 'compound_mssim':
        loss = compound_l1_mssim_loss
    model.compile(
        optimizer=Adam(lr=lr, **opt_kwargs),
        loss=loss,
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
    )

def compound_l1_mssim_loss(y_true, y_pred, alpha=0.84):
    mssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=tf.reduce_max(y_true))
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    loss = alpha * (1 - mssim) + (1 - alpha) * l1
    return loss
