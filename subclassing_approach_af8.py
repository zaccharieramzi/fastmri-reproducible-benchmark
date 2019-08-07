import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras_tqdm import TQDMCallback
import tensorflow as tf

from data import MaskedUntouched2DSequence, MaskedUntouchedSingleSlice2DSequence
from pdnet import PDNet, InvShiftCropNet
from utils import keras_psnr, keras_ssim




# paths
train_path = '/media/Zaccharie/UHRes/singlecoil_train/singlecoil_train/'
val_path = '/media/Zaccharie/UHRes/singlecoil_val/'
test_path = '/media/Zaccharie/UHRes/singlecoil_test/'




n_samples_train = 34742
n_samples_val = 7135

n_volumes_train = 973
n_volumes_val = 199




# generators
AF = 8
# MaskedUntouched2DSequence, MaskedUntouchedSingleSlice2DSequence
train_gen = MaskedUntouched2DSequence(train_path, af=AF, inner_slices=8)
val_gen = MaskedUntouched2DSequence(val_path, af=AF)




run_params = {
    'n_primal': 3,
    'n_dual': 3,
    'n_iter': 5,
    'n_filters': 16,
#     'n_primal': 2,
#     'n_dual': 2,
#     'n_iter': 2,
#     'n_filters': 8,
}

n_epochs = 20000
run_id = f'pdnet_subclassed_af{AF}_{int(time.time())}'
chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'




chkpt_cback = ModelCheckpoint(chkpt_path, period=500, save_weights_only=True)
log_dir = op.join('logs', run_id)
tboard_cback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
lr_on_plat_cback = ReduceLROnPlateau(monitor='val_loss', min_lr=5*1e-5, mode='auto', patience=3)
# fix from https://github.com/bstriner/keras-tqdm/issues/31#issuecomment-516325065
tqdm_cb = TQDMCallback(metric_format="{name}: {value:e}")
# tqdm_cb.on_train_batch_begin = tqdm_cb.on_batch_begin
# tqdm_cb.on_train_batch_end = tqdm_cb.on_batch_end
# tqdm_cb.on_test_begin = lambda x,y:None
# tqdm_cb.on_test_end = lambda x,y:None




# model = InvShiftCropNet(**run_params)
model = PDNet(**run_params)
model.compile(
    optimizer=Adam(lr=1e-2),
    loss='mean_absolute_error',
    metrics=['mean_squared_error', keras_psnr, keras_ssim],
)




model.fit_generator(
    train_gen,
    steps_per_epoch=10,
    epochs=n_epochs,
    validation_data=val_gen,
    validation_steps=25,
    verbose=2,
    callbacks=[tqdm_cb, tboard_cback, chkpt_cback, lr_on_plat_cback],
    max_queue_size=25,
#     use_multiprocessing=True,
    workers=25,
    shuffle=True,
)





# # overfitting trials

# data = train_gen[0]
# val_data = val_gen[0]
# model.fit(
#     x=data[0],
#     y=data[1],
# #     validation_data=val_data,
#     batch_size=data[0][0].shape[0],
#     epochs=100,
#     verbose=2,
#     shuffle=False,
# )
