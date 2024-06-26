"""
    Most borrow from: https://github.com/Alibaba-MIIL/ASL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.001, eps=1e-7, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, rec, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)

        self.xs_neg = 1 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets

                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                  self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                              self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                self.loss *= self.asymmetric_w
        _loss = -self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1)

        _loss += 0 * torch.sum(rec[0])
        _loss += 0 * torch.sum(rec[1])
        return _loss


class pretrainLossOptimized(nn.Module):
    def __init__(self, clip=0.05, eps=1e-5):
        super(pretrainLossOptimized, self).__init__()
        self.clip = clip
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None
        self.LARGE_NUM = 1e9

    def forward(self, ori, rec, hs):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        raw_h1: logits h1
        raw_h2: logits h2
        """

        x = ori[0]
        # print('x.shape=',x.shape)
        z = ori[1]
        rec_x = rec[0].squeeze()
        # print('rec_x.shape=',rec_x.shape)
        rec_z = rec[1].squeeze()

        recon_loss_1 = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(rec_x, x, reduction='none'), dim=1))
        recon_loss_2 = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(rec_z, z, reduction='none'), dim=1))
        recon_loss = recon_loss_1 + recon_loss_2
        recon_loss += 0 * torch.sum(hs)
        return recon_loss
