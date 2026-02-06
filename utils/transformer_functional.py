import torch


def patchify(x: torch.Tensor) -> torch.Tensor:
    b, k, c, h, w = x.shape
    x = x.permute(0, 1, 3, 4, 2) # [B, K, 128, H//4, W//4] -> [B, K, H//4, W//4, 128]
    patches = []
    for i in range(b):
        for j in range(k):
            src = x[i, j, :, :, :].unsqueeze(0) # [1, H//4, W//4, 128]
            tgt1 = x[i, :j, :, :, :] 
            tgt2 = x[i, j + 1:, :, :, :]
            tgt = torch.cat([tgt1, tgt2], dim=0) # [K - 1, H//4, W//4, 128]
            ex = torch.cat([src, tgt], dim=0) # [K=(src|tgt), H//4, W//4, 128]
            patches.append(ex)
    patches = torch.stack(patches, dim=0) # [B * K, K=(src|tgt), H//4, W//4, 128]
    return patches


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
