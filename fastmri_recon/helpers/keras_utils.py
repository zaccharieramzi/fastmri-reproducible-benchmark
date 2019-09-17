from keras.optimizers import Adam

from .utils import keras_psnr, keras_ssim


def default_model_compile(model, lr):
    model.compile(
        optimizer=Adam(lr=lr, clipnorm=1.),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
    )
