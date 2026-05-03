import torch
import torch.nn as nn


class SubjectViewFusion(nn.Module):
    """
    V1: subject-only view fusion.
    - img_views: [B, K, D]
    - subject_ids: [B]
    returns:
        fused: [B, D]
        weights: [B, K]
    """
    def __init__(self, max_subject_id: int, num_views: int):
        super().__init__()
        self.max_subject_id = max_subject_id
        self.num_views = num_views
        self.view_logits = nn.Embedding(max_subject_id + 1, num_views)
        nn.init.zeros_(self.view_logits.weight)

    def forward(self, img_views: torch.Tensor, subject_ids: torch.Tensor):
        if img_views.dim() != 3:
            raise ValueError(f"SubjectViewFusion expects img_views with shape [B, K, D], got {tuple(img_views.shape)}")

        subject_ids = subject_ids.long()
        logits = self.view_logits(subject_ids)     # [B, K]
        weights = torch.softmax(logits, dim=-1)    # [B, K]
        fused = torch.sum(weights.unsqueeze(-1) * img_views, dim=1)  # [B, D]
        return fused, weights
