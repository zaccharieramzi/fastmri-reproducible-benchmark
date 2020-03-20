import os.path as op
import random
import time

import tensorflow as tf
from keras_tqdm import TQDMCallback
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from fastmri_recon.data.sequences.oasis_sequences import Masked2DSequence
from fastmri_recon.models.functional_models.cascading import cascade_net


random.seed(0)

# paths
train_path = '/media/Zaccharie/UHRes/OASIS_data/'

# generators
AF = 4
train_gen = Masked2DSequence(
    train_path,
    af=AF,
    inner_slices=32,
    scale_factor=1e-2,
    seed=0,
    rand=True,
    val_split=0.1,
)
val_gen = train_gen.val_sequence
n_train = 1000
n_val = 200

train_gen.filenames = random.sample(train_gen.filenames, n_train)
val_gen.filenames = random.sample(val_gen.filenames, n_val)

run_params = {
    'n_cascade': 5,
    'n_convs': 5,
    'n_filters': 48,
    'noiseless': True,
}

n_epochs = 300
run_id = f'cascadenet_af{AF}_oasis_{int(time.time())}'
chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'

chkpt_cback = ModelCheckpoint(chkpt_path, period=100, save_weights_only=True)
log_dir = op.join('logs', run_id)
tboard_cback = TensorBoard(
    profile_batch=0,
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
tqdm_cb = TQDMCallback(metric_format="{name}: {value:e}")
tqdm_cb.on_train_batch_begin = tqdm_cb.on_batch_begin
tqdm_cb.on_train_batch_end = tqdm_cb.on_batch_end

model = cascade_net(input_size=(None, None, 1), fastmri=False, lr=1e-3, **run_params)

print(model.summary(line_length=150))

model.fit_generator(
    train_gen,
    steps_per_epoch=n_train,
    epochs=n_epochs,
    validation_data=val_gen,
    validation_steps=1,
    verbose=0,
    callbacks=[tqdm_cb, tboard_cback, chkpt_cback,],
    # max_queue_size=35,
    use_multiprocessing=True,
    workers=35,
    shuffle=False,
)
