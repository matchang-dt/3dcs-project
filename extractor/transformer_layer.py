import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def window_partition(x, window_size):
    # x: (n, h, w, c)
    if x.ndim == 4:
        n, h, w, c = x.shape
        x = x.view(
            n,
            h // window_size, window_size,
            w // window_size, window_size,
            c
        )
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() # [n, window_num_h, window_num_w, window_size, window_size, c]
        return windows.view(-1, window_size * window_size, c) # [n * window_num, window_size**2, c]
    else:
    # x: (n, k, h, w, c)
        n, k, h, w, c = x.shape
        x = x.view(
            n, k,
            h // window_size, window_size,
            w // window_size, window_size,
            c
        )
        windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous() # [n, window_num_h, window_num_w, k, window_size, window_size, c]
        return windows.view(-1, k * window_size * window_size, c) # [n * window_num, k * window_size**2, c]


def window_reverse(windows, window_size, h, w): ### to check
    # windows: (b * window_num, window_size**2, c)
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(
        b,
        h // window_size, # window_num_h
        w // window_size, # window_num_w
        window_size,
        window_size,
        -1, # c
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1) # [b, h, w, c]



def make_attn_mask(h, w, window_size, shift_size, device):
    img_mask = torch.zeros((1, h, w, 1), device=device)
    cnt = 0
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = h_slices
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size) # [1, window_num_h, window_num_w, window_size * window_size, 1]
    mask_windows = mask_windows.view(-1, window_size * window_size) # [window_num, window_size**2]
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None] # [window_num, window_size**2, window_size**2]
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")) # [window_num, window_size**2, window_size**2]
    return attn_mask # [window_num, window_size**2, window_size**2]


class WindowSelfAttention(L.LightningModule):
    def __init__(self, dim, window_size, num_heads=1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # relative position bias
        size = (2 * window_size - 1) ** 2
        self.rel_pos = nn.Parameter(torch.zeros(size, num_heads)) #[window_size*2 - 1)**2, num_heads]

        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))
        coords = coords.flatten(1) # [2, window_size**2]
        rel = coords[:, :, None] - coords[:, None, :] # [2, window_size**2, window_size**2]
        rel = rel.permute(1, 2, 0) # [window_size**2, window_size**2, 2]
        # This rel represents rel[i, j] = [x_i - x_j, y_i - y_j]
        rel[:, :, 0] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel[:, :, 1] += window_size - 1
        self.register_buffer("rel_index", rel.sum(-1)) # [window_size**2, window_size**2]

    def forward(self, x, mask=None):
        # x: [B*K*nW, window_size**2, 128]
        b, n, c = x.shape # b = B * nW, n = window_size**2
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads) # [B * nW, n, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B * nW, num_heads, n,head_dim]
        query, key, value = qkv # [B * nW, num_heads, n,head_dim]

        query = query * self.scale
        attn = query @ key.transpose(-2, -1) # [B * nW, num_heads, n, n]

        bias = self.rel_pos[self.rel_index.view(-1)] #[n, n, num_heads]
        bias = bias.view(n, n, -1).permute(2, 0, 1) # [num_heads, n, n]
        attn = attn + bias.unsqueeze(0) # [B * nW, num_heads, n, n]

        if mask is not None:
            nW = mask.shape[0]# mask: [nW, n, n]
            attn = attn.view(-1, nW, self.num_heads, n, n) # [B, nW, num_heads, n, n]
            attn = attn + mask.unsqueeze(1).unsqueeze(0) # [B, nW, num_heads, n, n]
            attn = attn.view(-1, self.num_heads, n, n) # [B*nW, num_heads, n, n]

        attn = attn.softmax(dim=-1) # [B * nW, num_heads, n, n]
        x = (attn @ value).transpose(1, 2).reshape(b, n, c) # c = num_heads * head_dim
        return self.proj(x) # [B*nW, n, c]


class WindowCrossAttention(L.LightningModule):
    def __init__(self, dim, window_size, num_heads=1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
        # relative position bias
        size = (2 * window_size - 1) ** 2
        self.rel_pos = nn.Parameter(torch.zeros(size, num_heads)) # [(2 * window_size - 1)**2, num_heads]

        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))
        coords = coords.flatten(1) # [2, window_size**2]
        rel = coords[:, :, None] - coords[:, None, :] # [2, window_size**2, window_size**2]
        rel = rel.permute(1, 2, 0) # [window_size**2, window_size**2, 2]
        # This rel represents rel[i, j] = [x_i - x_j, y_i - y_j]
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_index", rel.sum(-1)) # [window_size**2, window_size**2]

    def forward(self, x_q, x_kv, mask=None):
        # x_q: [B*K*nW, window_size**2, 128], x_kv: [B*K*nW, (K-1)*window_size**2, 128]
        b, n, c = x_q.shape # b = B * nW, n = window_size**2
        k = x_kv.shape[1] // n # k = K-1

        query = self.q(x_q).reshape(b, n, self.num_heads, c // self.num_heads) # [B * nW, n, num_heads, head_dim]
        kv = self.kv(x_kv).reshape(b, k * n, 2, self.num_heads, c // self.num_heads) # [B * nW, (K-1)*n, 2, num_heads, head_dim]
        query = query.permute(0, 2, 1, 3) # [B * nW, num_heads, n, head_dim]
        key, value = kv.permute(2, 0, 3, 1, 4) # [B * nW, num_heads, n*(K-1), head_dim]

        query = query * self.scale
        attn = query @ key.transpose(-2, -1) # [B * nW, num_heads, n, n*(K-1)]

        bias = self.rel_pos[self.rel_index.view(-1)] #[n, n, num_heads]
        bias = bias.view(n, n, -1).permute(2, 0, 1).repeat(1, 1, k) # [num_heads, n, (K-1)*n]
        attn = attn + bias.unsqueeze(0) # [B * nW, num_heads, n, n*(K-1)]

        if mask is not None:
            nW = mask.shape[0] # mask: [nW, n, n]
            mask = mask.repeat(1, 1, k) # [nW, n, (K-1)*n]
            attn = attn.view(-1, nW, self.num_heads, n, k * n) # [B, nW, num_heads, n, (K-1)*n]
            attn = attn + mask.unsqueeze(1).unsqueeze(0) # [B, nW, num_heads, n, (K-1)*n]
            attn = attn.view(-1, self.num_heads, n, k * n) # [B*nW, num_heads, n, (K-1)*n]

        attn = attn.softmax(dim=-1) # [B*nW, num_heads, n, n*(K-1)]
        x = (attn @ value).transpose(1, 2).reshape(b, n, c) # c = num_heads * head_dim
        return self.proj(x) # [B*nW, n, c]


class SwinCrossBlock(L.LightningModule):
    def __init__(self, dim=128, num_heads=1, window_size=32, shift_size=16):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.self_attn = WindowSelfAttention(dim, window_size, num_heads)
        self.cross_attn = WindowCrossAttention(dim, window_size, num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

        self.attn_mask = None

    def forward(self, x, x_kv):
        # x: (B*K, H//4, W//4, 128), x_kv: (B*K, K-1, H//4, W//4, 128)
        b, h, w, c = x.shape

        if self.shift_size > 0 and self.attn_mask is None:
            self.attn_mask = make_attn_mask(
                h, w, self.window_size, self.shift_size, x.device
            ) # [window_num, window_size**2, window_size**2]

        # Self Attention
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), (-2, -1)) # [B*K, H//4, W//4, 128]

        xw = window_partition(x, self.window_size) # [B*K*window_num, window_size**2, c]
        xw = self.self_attn(xw, self.attn_mask) # [B*K*window_num, window_size**2, c]

        x = window_reverse(xw, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (-2, -1))

        x = shortcut + x

        # Cross Attention
        shortcut = x
        x = self.norm_cross(x)
        x_kv = self.norm_cross(x_kv)

        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), (-2, -1))
            x_kv = torch.roll(x_kv, (-self.shift_size, -self.shift_size), (-2, -1))

        qw = window_partition(x, self.window_size)
        kvw = window_partition(x_kv, self.window_size) # [B*K*nW, (K-1)*window_size**2, 128]
        xw = self.cross_attn(qw, kvw, self.attn_mask)  # [B*K*nW, window_size**2, 128]

        x = window_reverse(xw, self.window_size, h, w) # [B*K, H//4, W//4, 128]

        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))

        x = shortcut + x # [B*K, H//4, W//4, 128]

        # FFN
        x = x + self.mlp(self.norm2(x)) # [B*K, H//4, W//4, 128]
        return x # [B*K, H//4, W//4, 128]
