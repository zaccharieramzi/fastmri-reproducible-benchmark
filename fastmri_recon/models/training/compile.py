from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import Adam

from ...evaluate.metrics.tf_metrics import keras_psnr, keras_ssim


def default_model_compile(model, lr):
    opt_kwargs = {}
    precision_policy = mixed_precision.global_policy()
    if precision_policy.loss_scale is None:
        opt_kwargs['clipnorm'] = 1.
    model.compile(
        optimizer=Adam(lr=lr, **opt_kwargs),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
    )
