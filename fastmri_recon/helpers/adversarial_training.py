import keras.callbacks as cbks
from keras.engine.training_utils import iter_sequence_infinite
from keras.optimizers import Adam
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.metrics_utils import to_list
import numpy as np

from .keras_utils import wasserstein_loss
from .utils import keras_ssim, keras_psnr


def compile_models(d, g, d_on_g, lr=1e-3, perceptual_weight=100, perceptual_loss='mse'):
    # strongly inspired by https://github.com/RaphaelMeudec/deblur-gan/blob/master/scripts/train.py#L26
    d_opt = Adam(lr=lr, clipnorm=1.)
    g_opt = Adam(lr=lr, clipnorm=1.)
    d_on_g_opt = Adam(lr=lr, clipnorm=1.)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    # to adjust with the typical mse (probably changing when dealing with normalized z_filled and kspaces scaled)
    loss_weights = [perceptual_weight, 1]
    generator_metrics = [keras_psnr, keras_ssim]
    if perceptual_loss != 'mse':
        generator_metrics.append('mse')
    discriminator_metrics = []
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights, metrics=[generator_metrics, discriminator_metrics])
    d.trainable = True
    # this because we want to evaluate only the output of the generator, and therefore will evaluate with it
    g.compile(optimizer=g_opt, loss=perceptual_loss, metrics=generator_metrics)

def adversarial_training_loop(g, d, d_on_g, train_gen, val_gen=None, validation_steps=1, n_epochs=1, n_batches=1, n_critic_updates=5, callbacks=None, workers=1, use_multiprocessing=False, max_queue_size=10):
    # all the gan stuff is from https://github.com/RaphaelMeudec/deblur-gan/blob/master/scripts/train.py#L26
    # NOTE: see if saving the weights of d_on_g is enough
    # Prepare display labels.
    out_labels = d_on_g.metrics_names
    # we only want to validate on the output of g
    val_out_labels = ['val_' + n for n in out_labels if g.name in n]
    callback_metrics = out_labels + val_out_labels

    # prepare callbacks
    # all the callback stuff is from https://github.com/keras-team/keras/blob/master/keras/engine/training_generator.py
    d_on_g.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=d_on_g.metrics_names[1:])]
    _callbacks += (callbacks or []) + [d_on_g.history]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    callback_model = d_on_g._get_callback_model()

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': n_epochs,
        'steps': n_batches,
        'verbose': 0,
        # 'do_validation': do_validation, to set when using validation data
        'metrics': callback_metrics,
    })
    callbacks._call_begin_hook('train')

    # all the queue stuff is from https://github.com/keras-team/keras/blob/master/keras/engine/training_generator.py
    if workers > 0:
        enqueuer = OrderedEnqueuer(
            train_gen,
            use_multiprocessing=use_multiprocessing,
            # TODO: add a parameter to control this
            shuffle=False,
        )
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        train_generator = enqueuer.get()
    else:
        train_generator = iter_sequence_infinite(train_gen)

    epoch_logs = {}
    d_losses = []
    for epoch in range(n_epochs):
        callbacks.on_epoch_begin(epoch)
        for batch_index in range(n_batches):
            x, image = next(train_generator)
            if isinstance(x, list):
                batch_size = len(x[0])
            else:
                batch_size = len(x)
            # build batch logs
            batch_logs = {'batch': batch_index, 'size': batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)
            output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

            generated_image = g.predict_on_batch(x)

            for _ in range(n_critic_updates):
                d_loss_real = d.train_on_batch(image, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_image, output_false_batch)
                # NOTE: this will not give a great loss unless we use what's underneath, we need to see
                # how to deal with this (maybe tensorboard won't be used, maybe we can use a custom callback)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            d.trainable = False

            outs = d_on_g.train_on_batch(x, [image, output_true_batch], reset_metrics=False)
            outs = to_list(outs)
            for l, o in zip(out_labels, outs):
                batch_logs[l] = o

            d.trainable = True

            callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)

        if val_gen:
            val_outs = g.evaluate_generator(
                val_gen,
                validation_steps,
                callbacks=callbacks,
                workers=0)
            val_outs = to_list(val_outs)
            # Same labels assumed.
            for l, o in zip(val_out_labels, val_outs):
                epoch_logs[l] = o
        callbacks.on_epoch_end(epoch, epoch_logs)
    callbacks._call_end_hook('train')
    return d_losses
