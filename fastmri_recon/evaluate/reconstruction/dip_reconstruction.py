import tensorflow as tf
from tensorflow.keras.losses import MSE

from fastmri_recon.models.subclassed_models.dip import DIPBase
from fastmri_recon.models.training.compile import default_model_compile

def dip_loss(y_true, y_pred):
    mse_real = MSE(tf.math.real(y_true), tf.math.real(y_pred))
    mse_imag = MSE(tf.math.imag(y_true), tf.math.imag(y_pred))
    mse_total = mse_real + mse_imag
    return mse_total

def reconstruct_dip(
        ktraj,
        kspace,
        model_checkpoint=None,
        save_model=False,
        save_path=None,
        in_dim=64,
        lr=1e-3,
        n_iter=10_000,
        multicoil=False,
        **model_kwargs,
    ):
    model = DIPBase(**model_kwargs)
    if model_checkpoint is not None:
        model.load_weights(model_checkpoint)
    default_model_compile(model, lr=lr, loss=dip_loss)
    in_random_vector = tf.random.normal([1, in_dim])
    _history = model.fit(
        x=[in_random_vector, ktraj],
        y=kspace,
        epochs=n_iter,
    )
    if save_model:
        model.save_weights(save_path)
    output = model.generate(in_random_vector, fastmri_format=True)
    return output
