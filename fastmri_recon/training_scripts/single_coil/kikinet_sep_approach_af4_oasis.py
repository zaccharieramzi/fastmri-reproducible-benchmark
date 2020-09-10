import os.path as op
import random
import time

from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.data.sequences.oasis_sequences import Masked2DSequence, KIKISequence
from fastmri_recon.models.functional_models.kiki_sep import kiki_sep_net
from fastmri_recon.models.utils.data_consistency import MultiplyScalar
from fastmri_recon.models.utils.non_linearities import lrelu

random.seed(0)


# paths
train_path = '/media/Zaccharie/UHRes/OASIS_data/'

n_train = 1000
n_val = 200


# generators
AF = 4
train_gen_last = Masked2DSequence(train_path, af=AF, inner_slices=32, rand=True, scale_factor=1e-2, seed=0, val_split=0.1)
val_gen_last = train_gen_last.val_sequence
train_gen_last.filenames = random.sample(train_gen_last.filenames, n_train)
val_gen_last.filenames = random.sample(val_gen_last.filenames, n_val)
random.seed(0)


train_gen_i = KIKISequence(train_path, af=AF, inner_slices=32, rand=True, scale_factor=1e-2, space='I', seed=0, val_split=0.1)
val_gen_i = train_gen_i.val_sequence
train_gen_i.filenames = random.sample(train_gen_i.filenames, n_train)
val_gen_i.filenames = random.sample(val_gen_i.filenames, n_val)
random.seed(0)

train_gen_k = KIKISequence(train_path, af=AF, inner_slices=32, rand=True, scale_factor=1e-2, space='K', seed=0, val_split=0.1)
val_gen_k = train_gen_k.val_sequence
train_gen_k.filenames = random.sample(train_gen_k.filenames, n_train)
val_gen_k.filenames = random.sample(val_gen_k.filenames, n_val)
random.seed(0)

run_params = {
    'n_convs': 16,
    'n_filters': 48,
    'noiseless': True,
    'lr': 1e-3,
    'activation': lrelu,
    'input_size': (None, None, 1),
}
multiply_scalar = MultiplyScalar()
n_epochs = 50

def learning_rate_from_epoch(epoch):
    return 10**(-(epoch // (n_epochs/3)) - 3)



def train_model(model, space='K', n=1):
    print(model.summary(line_length=150))
    run_id = f'kikinet_sep_{space}{n}_af{AF}_oasis_{int(time.time())}'
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
        steps_per_epoch=n_train,
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
model = kiki_sep_net(model, multiply_scalar, to_add='I', last=True, fastmri=False, **run_params)
train_model(model, space='I', n=2)
