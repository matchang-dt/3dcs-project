from math import sqrt

import torch
from torch import nn
import lightning as L

from .refiner import CostVolumeRefiner


def generate_volume_grids(h, w, depth_steps=128):
    """
    Generate the volume grids for the cost volume construction.
    Args:
        h (int): height of the feature map
        w (int): width of the feature map
        max_depth (float): maximum depth (far plane)
        depth_steps (int): number of depth steps
    Returns:
        volume_grids (torch.Tensor): output tensor of shape [h, w, d, 4]
    """
    # u_grids = ((2 * torch.arange(h) + 1) / h - 1).view(h, 1).expand(h, w)
    # v_grids = ((2 * torch.arange(w) + 1) / w - 1).view(1, w).expand(h, w)
    u_grids = torch.arange(h, dtype=torch.float32).reshape(h, 1).expand(h, w)
    v_grids = torch.arange(w, dtype=torch.float32).reshape(1, w).expand(h, w)
    uv_grids = torch.stack(
        [u_grids, v_grids, torch.ones_like(u_grids, dtype=torch.float32)], dim=2
    ).expand(h, w, depth_steps, 3) # [h, w, 128, 3]
    inv_depths = torch.arange(1, depth_steps + 1, dtype=torch.float32)
    inv_depths = inv_depths.reshape(1, 1, depth_steps, 1).expand(h, w, depth_steps, 1) #[h, w, 128, 1]
    volume_grids = torch.cat([uv_grids, inv_depths], dim=-1) # [h, w, 128, 4]
    return volume_grids # volume_grids[i, j, k] = [u, v, 1, 1/z]


def cost_volume_construct(P_src, P_tgt, f_src, f_tgt, volume_grids):
    """
    Construct the cost volume from the source and target features.
    Args:
        P_src (torch.Tensor): input tensor of shape [B, K, 4, 4] source camera projection matrix
        P_tgt (torch.Tensor): input tensor of shape [B, K, K - 1, 4, 4] target camera projection matrix
        f_src (torch.Tensor): input tensor of shape [B, K, h, w, c] source features
        f_tgt (torch.Tensor): input tensor of shape [B, K, (K - 1), h, w, c] target features
        volume_grids (torch.Tensor): input tensor of shape [h, w, d, 4] volume grids
        max_depth (float): maximum depth (far plane)
    Returns:
        cost_volume (torch.Tensor): output tensor of shape [B, K, h, w, d]
    """
    b, k, _, _, = P_src.shape
    h, w, d, _ = volume_grids.shape
    _, _, _, _, c = f_src.shape
    P_src_inv = torch.linalg.inv(P_tgt) # [B, K, K - 1, 4, 4]
    P_merged = torch.einsum('bklmn,bkno->bklmo', P_tgt, P_src_inv) # [B, K, K - 1, 4, 4]
    warped = torch.einsum('bklij,hwdj->bklhwdi', P_merged, volume_grids) # [B, K, K - 1, h, w, d, 4]: i = [uw, vw, w, w/z]
    warped_uv = warped[..., :2] / warped[..., 2] # [B, K, K - 1, h, w, d, 2]
    warped_uv = warped_uv.permute(0, 1, 2, 5, 3, 4, 6).view(b*k*(k-1)*d, h, w, d, 2) # [B*K*(K-1)*d, h, w, 2]
    f_src_reshaped = f_src.permute(0, 1, 4, 2, 3) # [B, K, c, h, w]
    f_tgt_reshaped = f_tgt.unsqueeze(-2).expand(b, k, k-1, h, w, d, c) # [B, K, K-1, h, w, d, c]
    f_tgt_reshaped = f_tgt_reshaped.permute(0, 1, 2, 5, 6, 3, 4).view(b*k*(k-1)*d, c, h, w) # [B*K*(K-1)*d, c, h, w]
    warped_features = torch.grid_sample(f_tgt_reshaped, warped_uv, mode='bilinear', padding_mode='zeros') # [B*K*(K-1)*d, c, h, w]
    warped_features = warped_features.view(b, k, k-1, d, c, h, w) # [B, K, K-1, d, c, h, w]
    cost_volume = torch.einsum('bkchw,bkldchw->bkdhw', f_src_reshaped, warped_features) # [B, K, d, h, w]
    cost_volume = cost_volume.permute(0, 1, 3, 4, 2) / sqrt(c) # [B, K, h, w, d]
    return cost_volume # [B, K, h, w, d]


class CostVolumeConstructor(L.LightningModule):
    """
    Cost volume constructor module.
    Constructs the cost volume from the source and target features and their projection matrices, then refines the cost volume with a U-net based refiner.
    """
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
        
        volume_grids = generate_volume_grids(h, w, depth_steps=feature_dim)
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
        # upsample the cost volume to the original image size
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
