from math import sqrt

import torch
from torch import nn
import lightning as L

from .refiner import CostVolumeRefiner

def generate_volume_grids(h, w, max_depth, depth_steps=128):
    u_grids = ((2 * torch.arange(h) + 1) / h - 1).view(h, 1).expand(h, w)
    v_grids = ((2 * torch.arange(w) + 1) / w - 1).view(1, w).expand(h, w)
    uv_grids = torch.stack(
        [u_grids, v_grids, torch.ones_like(u_grids)], dim=2
    ) # [h, w, 3]
    inv_depths = torch.arange(1, depth_steps + 1, dtype=torch.float32)
    depths = max_depth / inv_depths # [128]
    volume_grids = torch.einsum('hwc,d->hwdc', uv_grids, depths) # [h, w, 128, 3]
    volume_grids = torch.cat(
        [volume_grids, torch.ones_like(volume_grids[:, :, :, :1])], 
        dim=3,
    ) # [h, w, 128, 4]
    return volume_grids # volume_grids[i, j, k] = [u*d, v*d, d, 1]


def cost_volume_construct(P_src, P_tgt, f_src, f_tgt, volume_grids, max_depth):
    # P_src: [B, K, 4, 4]
    # P_tgt: [B, K, K - 1, 4, 4]
    # f_src: [B, K, h, w, c]
    # f_tgt: [B, K, (K - 1), h, w, c]
    # volume_grids: [h, w, d, 4]
    b, k, _, _, = P_src.shape
    h, w, d, _ = volume_grids.shape
    _, _, _, _, c = f_src.shape
    device = P_src.device
    eps = 1e-6
    f_src = f_src.unsqueeze(-2).expand(b, k, h, w, d, c) # [B, K, h, w, d, c]
    f_tgt = f_tgt.unsqueeze(-2).expand(b, k, k - 1, h, w, d, c) # [B, K, K - 1, h, w, d, c]
    P_tgt_inv = torch.linalg.inv(P_tgt) # [B, K, K - 1, 4, 4]
    P_merged = torch.einsum('bkmn,bklno->bklmo', P_src, P_tgt_inv) # [B, K, K - 1, 4, 4]
    warped = torch.einsum('bklij,hwdj->bklhwdi', P_merged, volume_grids) # [B, K, K - 1, h, w, d, 4]
    warped_uv = (warped[..., :2] / warped[..., 2:3] + 1) / 2 # [B, K, K - 1, h, w, d, 2]
    warped_ij = warped_uv * torch.tensor([h, w], device=device, dtype=warped_uv.dtype) - 0.5 # [B, K, K - 1, h, w, d, 2]
    warped_ij = warped_ij.round().long() # [B, K, K - 1, h, w, d, 2]
    warped_depth = warped[..., 2:3] # [B, K, K - 1, h, w, d, 1]
    warped_inv_depth = (max_depth / (warped_depth + eps)).round().long() # [B, K, K - 1, h, w, d, 1]
    warped_grids = torch.cat([warped_ij, warped_inv_depth - 1], dim=-1) # [B, K, K - 1, h, w, d, 3]
    grid_b = torch.arange(b, device=device).reshape(b, 1, 1, 1, 1, 1).expand(b, k, k - 1, h, w, d) # [B, K, K - 1, h, w, d]
    grid_k = torch.arange(k, device=device).reshape(1, k, 1, 1, 1, 1).expand(b, k, k - 1, h, w, d) # [B, K, K - 1, h, w, d]
    idx_i = warped_grids[..., 0] # [B, K, K - 1, h, w, d]
    idx_j = warped_grids[..., 1] # [B, K, K - 1, h, w, d]
    idx_d = warped_grids[..., 2] # [B, K, K - 1, h, w, d]
    mask = ((idx_i >= 0) & (idx_i < h) & (idx_j >= 0) & 
            (idx_j < w) & (idx_d >= 0) & (idx_d < d)) # [B, K, K - 1, h, w, d]
    indices = (
        grid_b[mask], grid_k[mask], idx_i[mask], idx_j[mask], idx_d[mask],
    ) # ([B * K * (K - 1) * h * w * d],) * 5
    f_tgt = f_tgt[mask] # [B * K * (K - 1) * h * w * d, c]

    warped_feat_values = torch.zeros(b, k, h, w, d, c, dtype=f_tgt.dtype, device=device)
    warped_feat_counts = torch.zeros(b, k, h, w, d, dtype=f_tgt.dtype, device=device)
    warped_feat = torch.zeros(b, k, h, w, d, c, dtype=f_tgt.dtype, device=device)
    warped_feat_values.index_put_(indices, f_tgt, accumulate=True) # [B, K, h, w, d, c]
    warped_feat_counts.index_put_(
        indices,
        torch.ones((indices[0].shape[0],), dtype=warped_feat_counts.dtype, device=device),
        accumulate=True,
    ) # [B, K, h, w, d]
    non_zero_mask = warped_feat_counts > 0
    warped_feat[non_zero_mask] = (
        warped_feat_values[non_zero_mask] / 
        warped_feat_counts[non_zero_mask].unsqueeze(-1)
    ) # [B, K, h, w, d, c]
    cost_volume = torch.einsum('bkhwdc,bkhwdc->bkhwd', f_src, warped_feat) # [B, K, h, w, d]
    cost_volume = cost_volume / sqrt(c) # [B, K, h, w, d]
    return cost_volume # [B, K, h, w, d]


class CostVolumeConstructor(L.LightningModule):
    def __init__(self, h, w, max_depth, feature_dim=128, dtype=torch.float32): # h=H//4, w=W//4
        # h = H//4, w = W//4
        super().__init__()
        self.max_depth = max_depth
        self.feature_dim = feature_dim
        group_num = feature_dim // 16
        
        self.refiner = CostVolumeRefiner(channels=feature_dim, feat_map_size=h, dtype=dtype)
        self.up_conv1 = nn.ConvTranspose2d(feature_dim, feature_dim, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.up_conv2 = nn.ConvTranspose2d(feature_dim, feature_dim, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.gn1 = nn.GroupNorm(num_groups=group_num, num_channels=feature_dim, eps=1e-6, dtype=dtype)
        self.gn2 = nn.GroupNorm(num_groups=group_num, num_channels=feature_dim, eps=1e-6, dtype=dtype)
        self.silu = nn.SiLU(inplace=True)
        self.last_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype)
        
        volume_grids = generate_volume_grids(h, w, max_depth, depth_steps=feature_dim)
        self.register_buffer('volume_grids', volume_grids)

    def forward(self, features, Ps):
        # features.shape  [B, K, H//4, W//4, 128]
        # Ps.shape [B, K, 4, 4]
        f_srcs = []
        f_tgts = []
        P_srcs = []
        P_tgts = []
        b, k, h, w, _ = features.shape
        for i in range(b):
            for j in range(k):
                f_src = features[i, j, :, :, :]
                P_src = Ps[i, j, :, :]
                f_tgt1 = features[i, :j, :, :, :]
                f_tgt2 = features[i, j + 1:, :, :, :]
                f_tgt = torch.cat([f_tgt1, f_tgt2], dim=0)
                P_tgt1 = Ps[i, :j, :, :]
                P_tgt2 = Ps[i, j + 1:, :, :]
                P_tgt = torch.cat([P_tgt1, P_tgt2], dim=0)
                f_srcs.append(f_src)
                f_tgts.append(f_tgt)
                P_srcs.append(P_src)
                P_tgts.append(P_tgt)
        f_srcs = torch.stack(f_srcs, dim=0).reshape(b, k, h, w, self.feature_dim)
        f_tgts = torch.stack(f_tgts, dim=0).reshape(b, k, k - 1, h, w, self.feature_dim)
        P_srcs = torch.stack(P_srcs, dim=0).reshape(b, k, 4, 4)
        P_tgts = torch.stack(P_tgts, dim=0).reshape(b, k, k - 1, 4, 4)
        cost_volumes = cost_volume_construct(
            P_srcs, 
            P_tgts, 
            f_srcs, 
            f_tgts, 
            self.volume_grids, 
            self.max_depth
        ) # [B, K, H//4, W//4, 128]
        refine_input = torch.cat([cost_volumes, features], dim=-1) # [B, K, H//4, W//4, 256]
        cost_volume_residuals = self.refiner(refine_input) # [B, K, H//4, W//4, 128]
        cost_volumes = cost_volumes + cost_volume_residuals # [B, K, H//4, W//4, 128]
        cost_volumes = cost_volumes.permute(0, 1, 4, 2, 3).reshape(b * k, self.feature_dim, h, w) # [B * K, 128, H//4, W//4]
        cost_volumes = self.up_conv1(cost_volumes) # [B * K, 128, H//2, W//2]
        cost_volumes = self.gn1(cost_volumes) # [B * K, 128, H//2, W//2]
        cost_volumes = self.silu(cost_volumes) # [B * K, 128, H//2, W//2]
        cost_volumes = self.up_conv2(cost_volumes) # [B * K, 128, H, W]
        cost_volumes = self.gn2(cost_volumes) # [B * K, 128, H, W]
        cost_volumes = self.silu(cost_volumes) # [B * K, 128, H, W]
        cost_volumes = self.last_conv(cost_volumes) # [B * K, 128, H, W]
        cost_volumes = cost_volumes.reshape(b, k, self.feature_dim, h*4, w*4).permute(0, 1, 3, 4, 2) # [B, K, H, W, 128]
        return cost_volumes # [B, K, H, W, 128]
