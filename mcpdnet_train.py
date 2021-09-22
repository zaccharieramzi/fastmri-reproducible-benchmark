import os.path as op
import random
import time

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow_addons.callbacks import TQDMProgressBar

from fastmri_recon.data.sequences.senior_mc_sequences import Masked_MC_2DSequence
from fastmri_recon.models.functional_models.mcpdnet import mcpdnet

random.seed(0)

import glob
# paths
train_path = '/neurospin/optimed/SeniorData/Multicontrast_trainval'

# generators
AF = 4
train_gen = Masked_MC_2DSequence(
    train_path,
    af=AF,
    inner_slices=32,
    scale_factor=1e-2,
    seed=0,
    rand=True,
    val_split=0.2,
)
val_gen = train_gen.val_sequence
n_train = 46
n_val = 11

train_gen.filenames = random.sample(train_gen.filenames, n_train)
val_gen.filenames = random.sample(val_gen.filenames, n_val)

print(train_gen.filenames)

from datetime import datetime

run_params = {
    'n_primal': 5,
    'n_dual': 5,
    'n_iter': 12,
    'n_filters': 32,
}

a = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
n_epochs = 50
run_id = f'mcpdnet_af{AF}_senior_{a}'
chkpt_path = f'checkpoints_senior/{run_id}' + '-{epoch:02d}.hdf5'
print(run_id)

chkpt_cback = ModelCheckpoint(chkpt_path, period=20, save_weights_only=True)
log_dir = op.join('logs', run_id)
tboard_cback = TensorBoard(
    profile_batch=0,
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
tqdm_cb = TQDMProgressBar()
lr_schedule = ReduceLROnPlateau(monitor='val_keras_psnr', factor=0.5, verbose=1, patience=15, mode='max'), 

## new_lr = lr * factor

model = mcpdnet(input_size=(None, None, None, 1), fastmri=False, lr=1e-3, **run_params)

print(model.summary(line_length=150))

history = model.fit(
    train_gen,
    steps_per_epoch=n_train,
    epochs=n_epochs,
    validation_data=val_gen,
    validation_steps=1,
    verbose=0,
    callbacks=[tqdm_cb, lr_schedule, tboard_cback, chkpt_cback,],
    # max_queue_size=35,
    use_multiprocessing=True,
    workers=35,
    shuffle=False,
)
import matplotlib.pyplot as plt
plt.figure()
fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.plot(history.history['keras_psnr'])
plt.plot(history.history['val_keras_psnr'])
plt.ylabel('PSNR (dB)')
plt.xlabel('Epochs')
plt.grid()
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

plt.figure()
fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.plot(history.history['keras_ssim'])
plt.plot(history.history['val_keras_ssim'])
#plt.title('Model Dice Metric')
plt.ylabel('SSIM')
plt.xlabel('Epochs')
plt.grid()
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Plot training & validation loss values
plt.figure()
fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Model Mean Squared Error')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.grid()
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.figure()
fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
