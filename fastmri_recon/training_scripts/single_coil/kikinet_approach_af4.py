import os.path as op
import time

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.data.sequences.fastmri_sequences import Masked2DSequence
from fastmri_recon.models.functional_models.kiki import kiki_net
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
train_gen = Masked2DSequence(train_path, af=AF, inner_slices=8, rand=True, scale_factor=1e6)
val_gen = Masked2DSequence(val_path, af=AF, scale_factor=1e6)





run_params = {
    'n_cascade': 2,
    'n_convs': 25,
    'n_filters': 32,
    'noiseless': True,
    'activation': lrelu,
}

n_epochs = 300
run_id = f'kikinet_af{AF}_{int(time.time())}'
chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
print(run_id)





chkpt_cback = ModelCheckpoint(chkpt_path, period=100, save_weights_only=True)
log_dir = op.join('logs', run_id)
tboard_cback = TensorBoard(
    profile_batch=0,
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
tqdm_cb = TQDMProgressBar()





model = kiki_net(lr=1e-3, **run_params)
print(model.summary(line_length=150))





model.fit_generator(
    train_gen,
    steps_per_epoch=n_volumes_train,
    epochs=n_epochs,
    validation_data=val_gen,
    validation_steps=1,
    verbose=0,
    callbacks=[tqdm_cb, tboard_cback, chkpt_cback,],
    # max_queue_size=35,
    use_multiprocessing=True,
    workers=35,
    shuffle=True,
)
