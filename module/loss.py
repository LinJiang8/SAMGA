import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute squared L2 distances between all pairs in x and y:
      dist2[i,j] = ||x_i - y_j||^2
    Uses (x^2 + y^2 - 2xy) trick for speed.
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)          # (B,1)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T        # (1,B)
    dist2 = x_norm + y_norm - 2.0 * (x @ y.T)           # (B,B)
    return dist2.clamp_min_(0.0)


def _mix_rbf_kernel(dist2: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """
    Multi-kernel RBF: K = mean_s exp(-dist2 / (2*sigma^2))
    dist2: (B,B)
    sigmas: (S,)
    """
    K = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * (s ** 2))
        K = K + torch.exp(-gamma * dist2)
    return K / sigmas.numel()

def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas=(0.1, 0.2, 0.5, 1.0, 2.0), unbiased: bool = True) -> torch.Tensor:
    """
    MMD^2 with (multi) RBF kernels.
    x, y: (B, D)
    Returns scalar.
    """
    assert x.dim() == 2 and y.dim() == 2, "x,y must be 2D tensors (B,D)"
    assert x.size(0) == y.size(0), "For this implementation, batch sizes must match"
    B = x.size(0)
    if B < 2:
        # Degenerate case: no meaningful unbiased estimate
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    sigmas_t = torch.tensor(sigmas, device=x.device, dtype=x.dtype)

    dist_xx = _pairwise_sq_dists(x, x)
    dist_yy = _pairwise_sq_dists(y, y)
    dist_xy = _pairwise_sq_dists(x, y)

    Kxx = _mix_rbf_kernel(dist_xx, sigmas_t)
    Kyy = _mix_rbf_kernel(dist_yy, sigmas_t)
    Kxy = _mix_rbf_kernel(dist_xy, sigmas_t)

    if unbiased:
        # remove diagonal terms for unbiased estimate
        sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (B * (B - 1))
        sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (B * (B - 1))
        sum_xy = Kxy.mean()
        mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    return mmd2.clamp_min(0.0)

class ContrastiveLoss(nn.Module):
    def __init__(self, init_temperature, alpha, beta, eeg_l2norm:bool, img_l2norm:bool, text_l2norm:bool, learnable:bool, is_softplus:bool):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eeg_l2norm = eeg_l2norm
        self.img_l2norm = img_l2norm
        self.text_l2norm = text_l2norm
        
        self.is_softplus = is_softplus
        
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature), requires_grad=learnable)
        self.softplus = nn.Softplus()

    def forward(self, eeg_feature, image_feature, text_feature):
        # L2 normalize embeddings
        if self.eeg_l2norm:
            eeg_feature = F.normalize(eeg_feature, p=2, dim=1)
        if self.img_l2norm:
            image_feature = F.normalize(image_feature, p=2, dim=1)
        if self.beta != 1.0:
            if self.text_l2norm:
                text_feature = F.normalize(text_feature, p=2, dim=1)

        # Calculate similarity matrix (N x N)
        if self.is_softplus:
            logit_scale = self.softplus(self.logit_scale)
        else:
            logit_scale = torch.exp(self.logit_scale)
        similarity_matrix_ie = torch.matmul(eeg_feature, image_feature.T) * logit_scale
        if self.beta != 1.0:
            similarity_matrix_te = torch.matmul(eeg_feature, text_feature.T) * logit_scale

        # Construct labels
        labels = torch.arange(eeg_feature.shape[0], device=eeg_feature.device)

        # Calculate two parts of the loss
        loss_eeg_ie = self.criterion_cls(similarity_matrix_ie, labels)
        loss_img_ie = self.criterion_cls(similarity_matrix_ie.T, labels)
        if self.beta != 1.0:
            loss_eeg_te = self.criterion_cls(similarity_matrix_te, labels)
            loss_img_te = self.criterion_cls(similarity_matrix_te.T, labels)
            
        if self.alpha != 1.0:
            loss_mse = self.criterion_mse(eeg_feature, image_feature)
        
        # Total loss is the average
        if self.beta != 1.0:
            loss_contrastive_ie = (loss_eeg_ie + loss_img_ie) / 2
            loss_contrastive_te = (loss_eeg_te + loss_img_te) / 2
            loss_contrastive = self.beta * loss_contrastive_ie + (1 - self.beta) * loss_contrastive_te
        else:
            loss_contrastive = (loss_eeg_ie + loss_img_ie) / 2
        
        if self.alpha != 1.0:
            loss = self.alpha * loss_contrastive + (1 - self.alpha) * loss_mse
        else:
            loss = loss_contrastive
        
        return loss