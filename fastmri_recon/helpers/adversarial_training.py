from keras.optimizers import Adam
import numpy as np

from .keras_utils import wasserstein_loss


def compile_models(d, d_on_g, lr=1e-3, perceptual_weight=100, perceptual_loss='mse'):
    d_opt = Adam(lr=lr, clipnorm=1.)
    d_on_g_opt = Adam(lr=lr, clipnorm=1.)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    # to adjust with the typical mse (probably changing when dealing with normalized z_filled and kspaces scaled)
    loss_weights = [perceptual_weight, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

def adversarial_training_loop(g, d, d_on_g, train_gen, n_epochs=1, n_batches=1, n_critic_updates=5):
    for epoch in range(n_epochs):
        for index in range(n_batches):
            # NOTE: add randomness in index
            # NOTE: when moving to cross domain, we need to add mask everywhere
            # and switch to kspace rather than z_filled images
    #         (kspace, mask), image = seq[index]
            z_filled_image, image = train_gen[index]
            batch_size = len(z_filled_image)
            output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

            generated_image = g.predict_on_batch(z_filled_image)

            for _ in range(n_critic_updates):
                d.train_on_batch(image, output_true_batch)
                d.train_on_batch(generated_image, output_false_batch)
                # NOTE: this will not give a great loss unless we use what's underneath, we need to see
                # how to deal with this (maybe tensorboard won't be used, maybe we can use a custom callback)
    #             d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
    #             d_losses.append(d_loss)

            d.trainable = False

            d_on_g.train_on_batch(z_filled_image, [image, output_true_batch])

            d.trainable = True

            # TODO: save weights of g
        # TODO: perform validation
