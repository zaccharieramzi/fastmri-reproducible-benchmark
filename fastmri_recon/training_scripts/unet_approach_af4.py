import os.path as op
import time

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras_tqdm import TQDMCallback

from data import ZeroFilled2DSequence
from unet import unet





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
train_gen = ZeroFilled2DSequence(train_path, af=AF, norm=True)
val_gen = ZeroFilled2DSequence(val_path, af=AF, norm=True)





run_params = {
    'n_layers': 4,
    'pool': 'max',
    "layers_n_channels": [16, 32, 64, 128],
    'layers_n_non_lins': 2,
#     'n_layers': 2,
}
n_epochs = 1
run_id = f'unet_wo_lastrelu_af{AF}_{int(time.time())}'
chkpt_path = f'checkpoints/{run_id}' + '-{epoch:02d}.hdf5'





chkpt_cback = ModelCheckpoint(chkpt_path, period=50)
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





model = unet(input_size=(320, 320, 1), lr=1e-3, **run_params)
print(model.summary())





model.fit_generator(
    train_gen,
    steps_per_epoch=n_volumes_train,
    epochs=n_epochs,
    validation_data=val_gen,
    validation_steps=n_volumes_val,
    verbose=0,
    callbacks=[tqdm_cb, tboard_cback, chkpt_cback, lr_on_plat_cback],
    max_queue_size=100,
    use_multiprocessing=True,
    workers=35,
)





# # overfitting trials

# data = train_gen[0]
# val_data = val_gen[0]
# model.fit(
#     x=data[0],
#     y=data[1],
# #     validation_data=val_data,
#     batch_size=data[0].shape[0],
#     epochs=50,
#     verbose=2,
#     shuffle=False,
# )
