import lightning as L
import torch
import torch.nn as nn

from .transformer_functional import window_partition, window_reverse, make_attn_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WindowSelfAttention(L.LightningModule):
    """
    Window-based self-attention.
    Using relative position bias (encoding), it is normal for SwinTransformer.
    It works as a normal self-attention block if the window size is the same as the input,
    except the positional encoding is relative.
    """
    def __init__(self, dim, window_size, num_heads=1, dtype=torch.float32):
        """
        Initialize the WindowSelfAttention.
        Args:
            dim (int): number of input channels
            window_size (int): size of the window
            num_heads (int): number of attention heads
            dtype (torch.dtype): data type
        """
        super().__init__()
        self.to(dtype)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, dtype=dtype)
        self.proj = nn.Linear(dim, dim, dtype=dtype)

        # relative position bias
        size = (2 * window_size - 1) ** 2
        self.rel_pos = nn.Parameter(torch.zeros(size, num_heads, dtype=dtype)) #[(window_size*2 - 1)**2, num_heads]

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
        """
        Forward pass of the WindowSelfAttention.
        Args:
            x (torch.Tensor): input tensor of shape [B*K*nW, window_size**2, 128]
            mask (torch.Tensor): mask tensor of shape [nW, window_size**2, window_size**2]
        Returns:
            out (torch.Tensor): output tensor of shape [B*K*nW, window_size**2, 128]
        """
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

        attn = attn.softmax(dim=-1).to(self.dtype) # [B * nW, num_heads, n, n]
        x = (attn @ value).transpose(1, 2).reshape(b, n, c) # c = num_heads * head_dim
        return self.proj(x) # [B*nW, n, c]


class WindowCrossAttention(L.LightningModule):
    """
    Window-based cross-attention.
    Using relative position bias (encoding), it is normal for SwinTransformer.
    It works as a normal cross-attention block if the window size is the same as the input,
    except the positional encoding is relative.
    """
    def __init__(self, dim, window_size, num_heads=1, dtype=torch.float32):
        """
        Initialize the WindowCrossAttention.
        Args:
            dim (int): number of input channels
            window_size (int): size of the window
            num_heads (int): number of attention heads
            dtype (torch.dtype): data type
        """
        super().__init__()
        self.to(dtype)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, dtype=dtype)
        self.kv = nn.Linear(dim, dim * 2, dtype=dtype)
        self.proj = nn.Linear(dim, dim, dtype=dtype)
        
        # relative position bias
        size = (2 * window_size - 1) ** 2
        self.rel_pos = nn.Parameter(torch.zeros(size, num_heads, dtype=dtype)) # [(2 * window_size - 1)**2, num_heads]

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
        """
        Forward pass of the WindowCrossAttention.
        Args:
            x_q (torch.Tensor): input source tensor of shape [B*K*nW, window_size**2, 128]
            x_kv (torch.Tensor): input target tensor of shape [B*K*nW, (K-1)*window_size**2, 128]
            mask (torch.Tensor): mask tensor of shape [nW, window_size**2, window_size**2]
        Returns:
            out (torch.Tensor): output tensor of shape [B*K*nW, window_size**2, 128]
        """
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

        attn = attn.softmax(dim=-1).to(self.dtype) # [B*nW, num_heads, n, n*(K-1)]
        x = (attn @ value).transpose(1, 2).reshape(b, n, c) # c = num_heads * head_dim
        return self.proj(x) # [B*nW, n, c]


class SwinCrossBlock(L.LightningModule):
    """
    SwinTransformer block with window-based self-attention and cross-attention.
    This is used in the feature extractor module.
    """
    def __init__(self, dim=128, num_heads=1, window_size=32, shift_size=16, dtype=torch.float32):
        """
        Initialize the SwinCrossBlock.
        The layer order is: self_attn->cross_attn->ffn + skip connection
        Args:
            dim (int): number of input channels
            num_heads (int): number of attention heads
            window_size (int): size of the window, basically the half of the image size
            shift_size (int): size of the shift, basically the half of the window size
            dtype (torch.dtype): data type
        """
        super().__init__()
        self.to(dtype)
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim, dtype=dtype)
        self.norm_cross = nn.LayerNorm(dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype)

        self.self_attn = WindowSelfAttention(dim, window_size, num_heads, dtype=dtype)
        self.cross_attn = WindowCrossAttention(dim, window_size, num_heads, dtype=dtype)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(4 * dim, dim, dtype=dtype),
        )

        self.attn_mask = None

    def forward(self, x, x_kv):
        """
        Forward pass of the SwinCrossBlock.
        Args:
            x (torch.Tensor): input tensor of shape [B*K, H//4, W//4, 128]
            x_kv (torch.Tensor): input tensor of shape [B*K, K-1, H//4, W//4, 128]
        Returns:
            out (torch.Tensor): output tensor of shape [B*K, H//4, W//4, 128]
        """
        # x: (B*K, H//4, W//4, 128), x_kv: (B*K, K-1, H//4, W//4, 128)
        b, h, w, c = x.shape

        if self.shift_size > 0 and self.attn_mask is None:
            self.attn_mask = make_attn_mask(
                h, w, self.window_size, self.shift_size, x.device,
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
