import tensorflow as tf
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

from fastmri_recon.models.subclassed_models.dip import DIPBase


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
        debug=False,
        output_shape=(320, 320),
        **model_kwargs,
    ):
    model = DIPBase(multicoil=multicoil, **model_kwargs)
    if model_checkpoint is not None:
        # build the model first before loading the weights
        in_random_vector = tf.random.normal([1, in_dim], seed=0 if debug else None)
        model([in_random_vector, ktraj[0:1]])
        model.load_weights(model_checkpoint)
    model.compile(
        optimizer=Adam(lr=lr),
        loss=dip_loss,
    )
    n_slices = kspace.shape[0]
    outputs = []
    for i_slice in range(n_slices):
        if i_slice == 1 and model_checkpoint is None:
            n_iter = n_iter // 10
        in_random_vector = tf.random.normal([1, in_dim], seed=0 if debug else None)
        _history = model.fit(
            x=[in_random_vector, ktraj[i_slice:i_slice+1]],
            y=kspace[i_slice:i_slice+1],
            epochs=n_iter,
            verbose=0,
        )
        if save_model:
            model.save_weights(save_path)
        output = model.generate(
            in_random_vector, 
            fastmri_format=True, 
            output_shape=output_shape,
        )
        outputs.append(output)
    output = tf.concat(outputs, axis=0)
    if debug:
        return output, _history
    else:
        return output
