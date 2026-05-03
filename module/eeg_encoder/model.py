import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import math

class SubjectAwareLayerMixer(nn.Module):
    """
    global layer prior + subject residual
    - global_logits: 全局层偏好，初始化时显式偏向 28 层
    - subject_bias: 每个 subject 的残差修正
    - 训练时 subject_dropout: 防止模型只记住 subject，保住 inter 能力
    """

    def __init__(
        self,
        layer_ids,
        num_subjects: int,
        prior_center: int = 28,
        prior_strength: float = 1.0,
        temperature: float = 1.0,
        subject_dropout: float = 0.3,
    ):
        super().__init__()
        self.layer_ids = list(layer_ids)
        self.num_layers = len(self.layer_ids)
        self.temperature = temperature
        self.subject_dropout = subject_dropout

        layer_ids_tensor = torch.tensor(self.layer_ids, dtype=torch.float32)
        self.register_buffer("layer_ids_tensor", layer_ids_tensor)

        if len(self.layer_ids) > 1:
            sorted_ids = sorted(self.layer_ids)
            diffs = [sorted_ids[i + 1] - sorted_ids[i] for i in range(len(sorted_ids) - 1)]
            positive_diffs = [d for d in diffs if d > 0]
            step = float(min(positive_diffs)) if len(positive_diffs) > 0 else 1.0
        else:
            step = 1.0

        # 显式让 28 层初始最高，越远权重越低
        # 例如 [20,24,28,32,36] -> logits 约为 [-2,-1,0,-1,-2] * prior_strength
        dist = torch.abs(layer_ids_tensor - float(prior_center)) / step
        init_logits = -prior_strength * dist

        self.global_logits = nn.Parameter(init_logits.clone())
        self.subject_bias = nn.Embedding(num_subjects, self.num_layers)
        nn.init.zeros_(self.subject_bias.weight)

    def forward(self, subject_ids: torch.Tensor = None, force_global: bool = False):
        """
        返回:
            weights: [B, K]
        """
        if subject_ids is None:
            logits = self.global_logits.unsqueeze(0)
        else:
            bsz = subject_ids.shape[0]
            logits = self.global_logits.unsqueeze(0).expand(bsz, -1)

            if not force_global:
                bias = self.subject_bias(subject_ids.long())

                if self.training and self.subject_dropout > 0:
                    keep_mask = (
                        torch.rand(bsz, 1, device=subject_ids.device) > self.subject_dropout
                    ).float()
                    bias = bias * keep_mask

                logits = logits + bias

        weights = torch.softmax(logits / self.temperature, dim=-1)
        return weights

    def get_global_weights(self):
        return torch.softmax(self.global_logits / self.temperature, dim=-1)

    def bias_reg(self):
        return self.subject_bias.weight.pow(2).mean()

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, D]
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = emb_size * expansion
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 4, dropout: float = 0.1, expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadSelfAttention(emb_size, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = FeedForward(emb_size, expansion=expansion, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int, emb_size: int, num_heads: int = 4, dropout: float = 0.1, expansion: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(emb_size, num_heads=num_heads, dropout=dropout, expansion=expansion)
            for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.layers:
            x = blk(x)
        return x


class TemporalPatchStem(nn.Module):
    """
    输入:  [B, C, T]
    输出:  [B, P, D]

    这里不再依赖固定 patch_size，也不手动切 patch，
    直接通过卷积提取时序模式，再用 AdaptiveAvgPool1d 对齐到固定 token 数 P。
    这样对 T=250 很稳，也不会出现 249/250 的错位问题。
    """
    def __init__(
        self,
        channels_num: int,
        emb_size: int,
        num_tokens: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels_num, emb_size, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Conv1d(emb_size, emb_size, kernel_size=9, padding=4, groups=emb_size, bias=False),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Conv1d(emb_size, emb_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(num_tokens),
        )

    def forward(self, x: Tensor) -> Tensor:
        # [B, C, T] -> [B, D, P] -> [B, P, D]
        x = self.stem(x)
        x = x.transpose(1, 2)
        return x


class SpatialChannelStem(nn.Module):
    """
    对每个 channel 各自做轻量时间编码，再把 channel 当 token。

    输入:  [B, C, T]
    输出:  [B, C, D]
    """
    def __init__(self, emb_size: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        b, c, t = x.shape
        x = x.reshape(b * c, 1, t)
        x = self.encoder(x)
        x = x.view(b, c, -1)
        return x


class ChannelAttentionPool(nn.Module):
    """
    对空间分支做 attention pooling，代替简单 mean。
    输入: [B, C, D]
    输出: [B, D]
    """
    def __init__(self, emb_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.score(x), dim=1)  # [B, C, 1]
        pooled = torch.sum(weights * x, dim=1)
        return pooled


class FusionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.Dropout(dropout),
            )),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class NeuroBridgeDBConformer(nn.Module):
    """
    适配 NeuroBridge 当前接口的双分支 EEG encoder

    输入:
        x: [B, C, T]
    输出:
        eeg_feature: [B, feature_dim]

    设计原则:
    1) 只改 EEG encoder，不碰后续 projector / SSP / CPA。
    2) 同一套结构同时兼容 17 通道和 63 通道。
    3) 保留 DBConformer 的双分支思想，但去掉分类头和不稳定的 gate。
    4) 明确输出 feature_dim，直接适配你现有 train.py。
    """
    def __init__(
        self,
        feature_dim: int = 1024,
        eeg_sample_points: int = 250,
        channels_num: int = 63,
        emb_size: int = 64,
        temporal_tokens: int = 10,
        temporal_depth: int = 3,
        spatial_depth: int = 3,
        num_heads: int = 4,
        spatial_hidden_dim: int = 32,
        dropout: float = 0.1,
        fusion_dropout: float = 0.3,
        use_temporal_cls: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.eeg_sample_points = eeg_sample_points
        self.channels_num = channels_num
        self.emb_size = emb_size
        self.temporal_tokens = temporal_tokens
        self.use_temporal_cls = use_temporal_cls

        # ---------- Temporal branch ----------
        self.temporal_stem = TemporalPatchStem(
            channels_num=channels_num,
            emb_size=emb_size,
            num_tokens=temporal_tokens,
            dropout=dropout,
        )
        temporal_token_count = temporal_tokens + (1 if use_temporal_cls else 0)
        self.temporal_pos = nn.Parameter(torch.randn(1, temporal_token_count, emb_size) * 0.02)
        if use_temporal_cls:
            self.temporal_cls = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        else:
            self.temporal_cls = None
        self.temporal_encoder = TransformerEncoder(
            depth=temporal_depth,
            emb_size=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            expansion=4,
        )

        # ---------- Spatial branch ----------
        self.spatial_stem = SpatialChannelStem(
            emb_size=emb_size,
            hidden_dim=spatial_hidden_dim,
            dropout=dropout,
        )
        self.spatial_pos = nn.Parameter(torch.randn(1, channels_num, emb_size) * 0.02)
        self.spatial_encoder = TransformerEncoder(
            depth=spatial_depth,
            emb_size=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            expansion=4,
        )
        self.spatial_pool = ChannelAttentionPool(emb_size)

        # ---------- Branch normalization ----------
        self.temporal_norm = nn.LayerNorm(emb_size)
        self.spatial_norm = nn.LayerNorm(emb_size)

        # ---------- Fusion head ----------
        self.fusion_head = FusionHead(input_dim=emb_size * 2, output_dim=feature_dim, dropout=fusion_dropout)

    def forward_temporal_branch(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        x_t = self.temporal_stem(x)  # [B, P, D]
        if self.use_temporal_cls:
            cls_tok = self.temporal_cls.expand(x_t.size(0), -1, -1)
            x_t = torch.cat([cls_tok, x_t], dim=1)  # [B, 1+P, D]
        x_t = x_t + self.temporal_pos[:, :x_t.size(1), :]
        x_t = self.temporal_encoder(x_t)
        if self.use_temporal_cls:
            x_t = x_t[:, 0]
        else:
            x_t = x_t.mean(dim=1)
        x_t = self.temporal_norm(x_t)
        return x_t

    def forward_spatial_branch(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        x_s = self.spatial_stem(x)  # [B, C, D]
        x_s = x_s + self.spatial_pos[:, :x_s.size(1), :]
        x_s = self.spatial_encoder(x_s)
        x_s = self.spatial_pool(x_s)  # [B, D]
        x_s = self.spatial_norm(x_s)
        return x_s

    def forward(self, x: Tensor) -> Tensor:
        # 兼容偶发输入 [B, 1, C, T]
        if x.dim() == 4:
            if x.size(1) != 1:
                raise ValueError(f"Expected x of shape [B,1,C,T] when 4D, but got {tuple(x.shape)}")
            x = x.squeeze(1)

        if x.dim() != 3:
            raise ValueError(f"Expected x of shape [B,C,T], but got {tuple(x.shape)}")

        b, c, t = x.shape
        if c != self.channels_num:
            raise ValueError(
                f"channels mismatch: model.channels_num={self.channels_num}, input channels={c}"
            )
        if t != self.eeg_sample_points:
            raise ValueError(
                f"time_points mismatch: model.eeg_sample_points={self.eeg_sample_points}, input time_points={t}"
            )

        x_t = self.forward_temporal_branch(x)  # [B, D]
        x_s = self.forward_spatial_branch(x)   # [B, D]
        x_fused = torch.cat([x_t, x_s], dim=-1)  # [B, 2D]
        out = self.fusion_head(x_fused)  # [B, feature_dim]
        return out

class EEGNet(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()

        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (channels_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                nn.Dropout2d(0.5)
            )
        
        # Use a dummy tensor to pass through the backbone to calculate the flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels_num, eeg_sample_points)
            out = self.backbone(dummy)
            embedding_dim = out.shape[1] * out.shape[2] * out.shape[3]
        
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.backbone(x)
        x = x.view(x.size(0), -1) 
        x = self.project(x)
        return x

class EEGProject(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        self.input_dim = eeg_sample_points * channels_num

        self.model = nn.Sequential(nn.Linear(self.input_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.3),
            )),
            nn.LayerNorm(feature_dim))
        
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class TSConv(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (channels_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        
        emb_size = 40
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))
        
        embedding_dim = (math.ceil((((eeg_sample_points - 25) + 1) - 51) / 5.) + 1) * 40
        self.proj_eeg = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.5),
            )),
            nn.LayerNorm(feature_dim),
        )
    
    def forward(self, x:Tensor):
        x = x.unsqueeze(dim=1)
        x = self.tsconv(x)
        x = self.projection(x)
        x = x.view(x.size(0), -1)
        x = self.proj_eeg(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class EEGTransformer(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        
        d_model = 128
        nhead = 8
        num_layers = 4
        dim_feedforward = 512
        dropout = 0.1
        
        # Project input (channels) -> embedding dimension
        self.input_proj = nn.Linear(channels_num, d_model)
        # Positional encoding across time dimension
        self.pos_encoder = PositionalEncoding(d_model, eeg_sample_points)
        # Transformer encoder (batch_first=True for [B, S, D])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Final projection to desired output dimension
        self.fc_out = nn.Linear(d_model, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG data tensor of shape [batch_size, channels_num, seq_len].
        Returns:
            Tensor of shape [batch_size, output_dim].
        """
        # Rearrange to [batch_size, seq_len, channels_num]
        x = x.permute(0, 2, 1)
        # Project to embedding dimension
        x = self.input_proj(x)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Transformer encoding
        x = self.transformer_encoder(x)
        # Pool across time (mean pooling)
        x = x.mean(dim=1)  # [batch_size, d_model]
        # Final feature projection
        x = self.fc_out(x)  # [batch_size, output_dim]
        return x


if __name__ == "__main__":
    # Example usage
    eeg_sample_points = 250
    channels_num = 17
    feature_dim = 1024
    model = EEGTransformer(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    
    # Create a dummy EEG input tensor with shape (batch_size, channels_num, eeg_sample_points)
    batch_size = 8
    dummy_eeg_input = torch.randn(batch_size, channels_num, eeg_sample_points)
    
    # Forward pass through the model
    output = model(dummy_eeg_input)
    print(output.shape)  # Expected output shape: (batch_size, feature_dim)