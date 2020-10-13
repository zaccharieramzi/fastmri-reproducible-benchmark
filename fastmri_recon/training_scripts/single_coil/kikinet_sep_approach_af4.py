import os.path as op
import time

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.data.sequences.fastmri_sequences import Masked2DSequence, KIKISequence
from fastmri_recon.models.functional_models.kiki_sep import kiki_sep_net
from fastmri_recon.models.utils.data_consistency import MultiplyScalar
from fastmri_recon.models.utils.non_linearities import lrelu


# paths
train_path = '/media/Zaccharie/UHRes/singlecoil_train/singlecoil_train/'
val_path = '/media/Zaccharie/UHRes/singlecoil_val/'
test_path = '/media/Zaccharie/UHRes/singlecoil_test/'

n_samples_train = 34742
n_samples_val = 7135

n_volumes_train = 973
n_volumes_val = 199


# generators
AF = 4
train_gen_last = Masked2DSequence(train_path, af=AF, inner_slices=8, rand=True, scale_factor=1e6)
val_gen_last = Masked2DSequence(val_path, af=AF, scale_factor=1e6)
train_gen_i = KIKISequence(train_path, af=AF, inner_slices=8, rand=True, scale_factor=1e6, space='I')
val_gen_i = KIKISequence(val_path, af=AF, scale_factor=1e6, space='I')
train_gen_k = KIKISequence(train_path, af=AF, inner_slices=8, rand=True, scale_factor=1e6, space='K')
val_gen_k = KIKISequence(val_path, af=AF, scale_factor=1e6, space='K')

run_params = {
    'n_convs': 25,
    'n_filters': 48,
    'noiseless': True,
    'lr': 1e-3,
    'activation': lrelu,
}
multiply_scalar = MultiplyScalar()
n_epochs = 50

def learning_rate_from_epoch(epoch):
    return 10**(-(epoch // (n_epochs/3)) - 3)



def train_model(model, space='K', n=1):
    print(model.summary(line_length=150))
    run_id = f'kikinet_sep_{space}{n}_af{AF}_{int(time.time())}'
    chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)

    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs//2)
    log_dir = op.join('logs', run_id)
    tboard_cback = TensorBoard(
        profile_batch=0,
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )
    lrate_cback = LearningRateScheduler(learning_rate_from_epoch)
    tqdm_cb = TQDMProgressBar()
    if space == 'K':
        train_gen = train_gen_k
        val_gen = val_gen_k
    elif space == 'I':
        if n == 2:
            train_gen = train_gen_last
            val_gen = val_gen_last
        elif n == 1:
            train_gen = train_gen_i
            val_gen = val_gen_i
    model.fit_generator(
        train_gen,
        steps_per_epoch=n_volumes_train,
        epochs=n_epochs,
        validation_data=val_gen,
        validation_steps=1,
        verbose=0,
        callbacks=[tqdm_cb, tboard_cback, chkpt_cback, lrate_cback,],
        # max_queue_size=35,
        use_multiprocessing=True,
        workers=35,
        shuffle=True,
    )
    return model

# first K net training
model = kiki_sep_net(None, multiply_scalar, to_add='K', last=False, **run_params)
train_model(model, space='K', n=1)
model = kiki_sep_net(model, multiply_scalar, to_add='I', last=False, **run_params)
train_model(model, space='I', n=1)
model = kiki_sep_net(model, multiply_scalar, to_add='K', last=False, **run_params)
train_model(model, space='K', n=2)
model = kiki_sep_net(model, multiply_scalar, to_add='I', last=True, **run_params)
train_model(model, space='I', n=2)
