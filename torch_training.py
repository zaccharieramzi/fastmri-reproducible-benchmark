import time

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from tqdm import tqdm


def torch_psnr(image_pred, image_gt):
    mse = F.mse_loss(image_pred, image_gt, reduction='mean')
    psnr = 10 * torch.log10(torch.max(image_gt)**2 / mse)
    return psnr


def train_epoch(epoch, model, data_loader, optimizer, writer, device, hard_limit=None, tqdm_wrapper=tqdm):
    model.train()
    global_step = epoch * len(data_loader)
    for i_iter, data in tqdm_wrapper(enumerate(data_loader), total=len(data_loader), desc='Training iterations'):
        if hard_limit is not None and i_iter > hard_limit:
            break
        kspace, mask, image_gt = data
        kspace = kspace[0]
        mask = mask[0]
        image_gt = image_gt[0]
        kspace = kspace.to(device)
        mask = mask.to(device)
        image_gt = image_gt.to(device)
        image_pred = model(kspace, mask)


        loss = F.l1_loss(image_pred, image_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if writer is not None:
            writer.add_scalar('TrainLoss', loss.item(), global_step + i_iter)
            writer.add_scalar('TrainPSNR', torch_psnr(image_pred, image_gt), global_step + i_iter)


def evaluate(epoch, model, data_loader, writer, device, hard_limit=None, tqdm_wrapper=tqdm):
    def save_images(image, tag):
        image -= image.min()
        image /= image.max()
        image = image.unsqueeze(1)
        writer.add_images(tag, image, epoch)

    model.eval()
    losses = []
    psnrs = []
    with torch.no_grad():
        for i_iter, data in tqdm_wrapper(enumerate(data_loader), total=len(data_loader), desc='Validation iterations'):
            if hard_limit is not None and i_iter > hard_limit:
                break
            kspace, mask, image_gt = data
            kspace = kspace[0]
            mask = mask[0]
            image_gt = image_gt[0]
            kspace = kspace.to(device)
            mask = mask.to(device)
            image_gt = image_gt.to(device)
            image_pred = model(kspace, mask)

            loss = F.mse_loss(image_pred, image_gt, reduction='mean')
            losses.append(loss.item())
            psnr = torch_psnr(image_pred, image_pred)
            psnrs.append(psnr.item())
        if writer is not None:
            save_images(image_gt, 'Target')
            save_images(image_pred, 'Reconstruction')
            save_images(torch.abs(image_gt - image_pred), 'Error')
            writer.add_scalar('ValMSE', np.mean(losses), epoch)
            writer.add_scalar('ValPSNR', np.mean(psnrs), epoch)



def save_model(chkpt_path, run_id, model):
    torch.save(
        {
            'model': model.state_dict(),
            'chkpt_path': chkpt_path,
        },
        f=chkpt_path / f'{run_id}.pt'
    )


def fit_torch(model, train_loader, val_loader, epochs, writer, optimizer, chkpt_path, run_id=None, device='cpu', save_freq=100, hard_limit_train=None, hard_limit_val=None, tqdm_wrapper=tqdm):
    if run_id is None:
        run_id = str(int(time.time()))
    # dummy_kspace = torch.randn(1, 640, 422, 2, device=device)
    # dummy_mask = torch.randn(1, 640, 422, device=device)
    # if writer is not None:
    #     writer.add_graph(model, [dummy_kspace, dummy_mask])
    for epoch in tqdm_wrapper(range(epochs), total=epochs, desc='Epochs'):
        train_epoch(epoch, model, train_loader, optimizer, writer, device, hard_limit=hard_limit_train, tqdm_wrapper=tqdm_wrapper)
        if val_loader is not None:
            evaluate(epoch, model, val_loader, writer, device, hard_limit=hard_limit_val, tqdm_wrapper=tqdm_wrapper)
        if (epoch + 1) % save_freq == 0:
            save_model(chkpt_path, run_id, model)
    if writer:
        writer.close()
