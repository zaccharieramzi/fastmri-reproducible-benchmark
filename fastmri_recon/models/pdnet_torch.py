"""Learned Primal-dual network adapted to MRI."""
import torch
from torch import nn

from ..helpers.torch_utils import ConvBlock
from ..helpers.transforms import ifft2, fft2, center_crop, complex_abs


class PDNet(torch.nn.Module):
    def __init__(self, n_filters=32, n_primal=5, n_dual=5, n_iter=10, primal_only=False):
        super(PDNet, self).__init__()
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.n_iter = n_iter
        self.n_filters = n_filters
        self.primal_only = primal_only

        self.primal_conv_layers = nn.ModuleList([ConvBlock(3, n_filters, 2 * (n_primal + 1), 2 * n_primal) for _ in range(n_iter)])
        if not self.primal_only:
            self.dual_conv_layers = nn.ModuleList([ConvBlock(3, n_filters, 2 * (n_dual + 2), 2 * n_dual) for _ in range(n_iter)])


    def forward(self, kspace, mask):
        mask = mask[..., None]
        mask = mask.expand_as(kspace).float()
        primal = torch.stack([torch.zeros_like(kspace)] * self.n_primal, dim=-1)
        if not self.primal_only:
            dual = torch.stack([torch.zeros_like(kspace)] * self.n_dual, dim=-1)

        for i, primal_conv_layer in enumerate(self.primal_conv_layers):
            dual_eval_exp = fft2(primal[..., 1])
            dual_eval_exp = dual_eval_exp * mask
            if self.primal_only:
                dual = dual_eval_exp - kspace
            else:
                update = torch.cat([dual[:, :, :, 0], dual[:, :, :, 1], dual_eval_exp, kspace], axis=-1)
                update = update.permute(0, 3, 1, 2)
                update = self.dual_conv_layers[i](update)
                update = update.permute(0, 2, 3, 1)
                update = torch.stack([update[..., :self.n_dual], update[..., self.n_dual:]], dim=-1)
                update = update.permute(0, 1, 2, 4, 3)
                dual = dual + update

            primal_exp = ifft2(mask * dual[..., 0])
            update = torch.cat([primal[:, :, :, 0], primal[:, :, :, 1], primal_exp], axis=-1)
            update = update.permute(0, 3, 1, 2)
            update = primal_conv_layer(update)
            update = update.permute(0, 2, 3, 1)
            update = torch.stack([update[..., :self.n_primal], update[..., self.n_primal:]], dim=-1)
            update = update.permute(0, 1, 2, 4, 3)
            primal = primal + update


        image = primal[..., 0]
        # equivalent of taking the module of the image
        image = complex_abs(image)
        image = center_crop(image, (320, 320))
        return image
