import numpy as np
import tensorflow as tf
import tensorflow.keras.callbacks as cbks
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import OrderedEnqueuer, GeneratorEnqueuer
from tensorflow.python.keras.callbacks import CallbackList

from .loss import wasserstein_loss
# from .metrics import mean_output, discriminator_accuracy
from ..keras_utils import iter_sequence_infinite, is_sequence, to_list
# from ....evaluate.metrics.tf_metrics import keras_ssim, keras_psnr


def _replace_label_first_underscore(label):
    label = label.replace('_', '/', 1)
    return label

def prepare_callbacks(g, d, callbacks, n_epochs=1, n_batches=1, include_d_metrics=False):
    # all the callback stuff is from https://github.com/keras-team/keras/blob/master/keras/engine/training_generator.py
    # NOTE: see if saving the weights of d_on_g is enough
    # Prepare display labels.
    out_labels = g.metrics_names
    out_labels = [_replace_label_first_underscore(l) for l in out_labels]
    # we only want to validate on the output of g
    val_out_labels = ['val_' + n for n in out_labels if g.name in n]
    callback_metrics = out_labels + val_out_labels
    if include_d_metrics:
        d_metrics_names = d.metrics_names
        d_metrics_fake = ['d_training/' + l + '_fake' for l in d_metrics_names]
        d_metrics_real = ['d_training/' + l + '_real' for l in d_metrics_names]
        d_metrics_names = d_metrics_fake + d_metrics_real
        callback_metrics += d_metrics_names
    # prepare callbacks
    g.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=g.metrics_names[1:])]
    _callbacks += (callbacks or []) + [g.history]
    callbacks = CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    callback_model = g._get_callback_model()

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': n_epochs,
        'steps': n_batches,
        'verbose': 0,
        # 'do_validation': do_validation, to set when using validation data
        'metrics': callback_metrics,
    })
    if not include_d_metrics:
        d_metrics_fake, d_metrics_real = None, None
    return callbacks, out_labels, val_out_labels, d_metrics_fake, d_metrics_real

def queue_train_generator(train_gen, workers=1, use_multiprocessing=False, max_queue_size=10, use_sequence_api=True):
    # all the queue stuff is from https://github.com/keras-team/keras/blob/master/keras/engine/training_generator.py
    if workers > 0:
        if use_sequence_api:
            enqueuer = OrderedEnqueuer(
                train_gen,
                use_multiprocessing=use_multiprocessing,
                # TODO: add a parameter to control this
                shuffle=False,
            )
        else:
            enqueuer = GeneratorEnqueuer(
                train_gen,
                use_multiprocessing=use_multiprocessing,
            )
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        train_generator = enqueuer.get()
    else:
        if use_sequence_api:
            train_generator = iter_sequence_infinite(train_gen)
        else:
            train_generator = train_gen
    return train_generator

def fill_batch_logs_w_d_metrics(batch_logs, d_outs_fake, d_outs_real, d_metrics_fake, d_metrics_real):
    d_outs_fake = np.array(d_outs_fake)
    d_outs_real = np.array(d_outs_real)
    for i, l in enumerate(d_metrics_fake):
        batch_logs[l] = np.mean(d_outs_fake[:, i])
    for i, l in enumerate(d_metrics_real):
        batch_logs[l] = np.mean(d_outs_real[:, i])

@tf.function
def train_step(
    x, image, g, d, g_opt, d_opt, n_critic_updates, perceptual_weight=100, perceptual_loss='mse'
):
    for _ in range(n_critic_updates):
        with tf.GradientTape() as tape:
            predictions_real = d(image)
            d_loss_real = wasserstein_loss(
                tf.ones_like(predictions_real), predictions_real
            )

            generated_images = g(x)
            predictions_fake = d(generated_images)
            d_loss_fake = wasserstein_loss(
                -tf.ones_like(predictions_fake), predictions_fake
            )

            d_loss = tf.math.reduce_mean(0.5 * tf.math.add(d_loss_real, d_loss_fake))

        gradients = tape.gradient(d_loss, d.trainable_weights)
        d_opt.apply_gradients(zip(gradients, d.trainable_weights))

    with tf.GradientTape() as tape:
        generated_images = g(x)
        predictions = d(generated_images)

        discriminator_loss = wasserstein_loss(tf.ones_like(predictions), predictions)

        image_loss = tf.losses.get(perceptual_loss)(image, generated_images)

        g_loss = tf.math.reduce_mean(perceptual_weight * image_loss + discriminator_loss)

    gradients = tape.gradient(g_loss, g.trainable_weights)
    g_opt.apply_gradients(zip(gradients, g.trainable_weights))

    return g_loss, d_loss

def adversarial_training_loop(
        g,
        d,
        train_gen,
        d_lr=1e-3,
        g_lr=1e-3,
        val_gen=None,
        validation_steps=1,
        n_epochs=1,
        n_batches=1,
        n_critic_updates=5,
        callbacks=None,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        include_d_metrics=False,
        perceptual_weight=1,
        perceptual_loss='mse',
    ):
    d_opt = RMSprop(lr=d_lr, clipnorm=1.)
    g_opt = Adam(lr=g_lr, clipnorm=1.)
    # all the gan stuff is from https://github.com/RaphaelMeudec/deblur-gan/blob/master/scripts/train.py#L26
    callbacks, out_labels, val_out_labels, d_metrics_fake, d_metrics_real = prepare_callbacks(
        g,
        d,
        callbacks,
        n_epochs=n_epochs,
        n_batches=n_batches,
        include_d_metrics=False,
    )
    callbacks._call_begin_hook('train')
    use_sequence_api = is_sequence(train_gen)
    train_generator = queue_train_generator(
        train_gen,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
        use_sequence_api=use_sequence_api,
    )

    epoch_logs = {}
    generator_metric = tf.keras.metrics.Mean()
    discriminator_metric = tf.keras.metrics.Mean()
    for epoch in range(n_epochs):
        callbacks.on_epoch_begin(epoch)
        for batch_index in range(n_batches):
            x, image = next(train_generator)
            if isinstance(x, list):
                batch_size = len(x[0])
            else:
                batch_size = len(x)
            x = tf.convert_to_tensor(x)
            # TODO: handle case where x is a list (i.e. image + mask)
            image = tf.convert_to_tensor(image)
            # build batch logs
            batch_logs = {'batch': batch_index, 'size': batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)
            g_loss, d_loss = train_step(x, image, g, d, g_opt, d_opt, n_critic_updates, perceptual_weight=perceptual_weight, perceptual_loss=perceptual_loss)
            # TODO: find a way to include discriminator_metrics
            # if include_d_metrics:
                    # d_outs_fake.append(d_out_fake)
                    # d_outs_real.append(d_out_real)
            # if include_d_metrics:
                # fill_batch_logs_w_d_metrics(
                    # batch_logs,
                    # d_outs_fake,
                    # d_outs_real,
                    # d_metrics_fake,
                    # d_metrics_real,
                # )
            generator_metric(g_loss)
            discriminator_metric(d_loss)

            callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)

        if val_gen:
            val_outs = g.evaluate(
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
