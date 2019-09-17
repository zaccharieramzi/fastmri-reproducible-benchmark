import os.path as op
import time

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fastmri_recon.data.fastmri_datasets import Masked2DDataset
from fastmri_recon.models.pdnet import PDNet
from fastmri_recon.helpers.torch_training import fit_torch



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
train_gen = Masked2DDataset(train_path, af=AF, inner_slices=8, rand=True, scale_factor=1e6)
val_gen = Masked2DDataset(val_path, af=AF, scale_factor=1e6)





run_params = {
    'n_primal': 5,
    'n_dual': 5,
    'n_iter': 10,
    'n_filters': 32,
}

n_epochs = 300
run_id = f'pdnet_torch_af{AF}_{int(time.time())}'
chkpt_path = 'checkpoints'





log_dir = op.join('logs', run_id)
model = PDNet(**run_params)
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, n_epochs)
writer = SummaryWriter(log_dir=log_dir)

model.cuda()


val_gen.filenames = val_gen.filenames[:5]
train_loader = DataLoader(
    dataset=train_gen,
    batch_size=1,
    shuffle=False,
    num_workers=35,
    pin_memory=True,
)
val_loader = DataLoader(
    dataset=val_gen,
    batch_size=1,
    num_workers=5,
    pin_memory=True,
    shuffle=False,
)
fit_torch(
    model,
    train_loader,
    val_loader,
    n_epochs,
    writer,
    optimizer,
    chkpt_path,
    run_id=run_id,
    device='cuda',
    save_freq=100,
    scheduler=scheduler
)
