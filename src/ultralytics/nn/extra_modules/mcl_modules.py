"""MCL-DETR modules and ablation controls used by this release only."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv, RepConv
from ultralytics.nn.modules.transformer import AIFI, TransformerEncoderLayer


class LayerNorm(nn.Module):
    """Channel-first LayerNorm for feature maps."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSPConv(nn.Module):
    """Cascaded partial depthwise convolution used by the MSPC backbone."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv(channels, channels, k=3)
        self.conv2 = Conv(channels // 2, channels // 2, k=5, g=channels // 2)
        self.conv3 = Conv(channels // 4, channels // 4, k=7, g=channels // 4)
        self.conv4 = Conv(channels, channels, 1)

    def forward(self, x):
        first, bypass = self.conv1(x).chunk(2, dim=1)
        second = self.conv2(first)
        third, retained = second.chunk(2, dim=1)
        return self.conv4(torch.cat([self.conv3(third), retained, bypass], dim=1)) + x


class MSPC(C2f):
    """Cross-stage partial wrapper for the multi-scale partial convolution unit."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MSPConv(self.c) for _ in range(n))


class RepNBottleneck(nn.Module):
    """RepNCSPELAN4 bottleneck retained for the MCAF ablation control."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = RepConv(c1, hidden)
        self.cv2 = Conv(hidden, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class RepNCSP(nn.Module):
    """RepNCSPELAN4-compatible CSP block."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1)
        self.m = nn.Sequential(*(RepNBottleneck(hidden, hidden, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    """Re-parameterized fusion block used by the no-CAA control."""

    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, 1))


class CAA(nn.Module):
    """Context anchor attention with a 7x7 pool and separable 1x11/11x1 paths."""

    def __init__(self, channels, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(channels, channels)
        self.h_conv = nn.Conv2d(channels, channels, (1, h_kernel_size), 1,
                                (0, h_kernel_size // 2), 1, channels)
        self.v_conv = nn.Conv2d(channels, channels, (v_kernel_size, 1), 1,
                                (v_kernel_size // 2, 0), 1, channels)
        self.conv2 = Conv(channels, channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attention = self.avg_pool(x)
        attention = self.conv2(self.v_conv(self.h_conv(self.conv1(attention))))
        return self.act(attention) * x


class MCAF(RepNCSPELAN4):
    """RepNCSPELAN4 fusion followed by context anchor attention."""

    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__(c1, c2, c3, c4, c5)
        self.caa = CAA(c3 + 2 * c4)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(self.caa(torch.cat(y, 1)))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(self.caa(torch.cat(y, 1)))


class LRPB(nn.Module):
    """Continuous learnable relative-position bias generated from coordinate offsets."""

    def __init__(self, dim, num_heads, residual=False):
        super().__init__()
        self.residual = residual
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, num_heads))

    def forward(self, offsets):
        position = self.pos_proj(offsets)
        if self.residual:
            position = position + self.pos1(position)
            position = position + self.pos2(position)
            return self.pos3(position)
        return self.pos3(self.pos2(self.pos1(position)))


class LRPB_Attention(nn.Module):
    """Self-attention augmented by resolution-adaptive continuous relative bias."""

    def __init__(self, dim, group_size=None, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, position_bias=True):
        super().__init__()
        self.group_size = group_size
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.position_bias = position_bias
        if position_bias:
            self.pos = LRPB(dim // 4, num_heads)
            self._bias_table_cache = {}
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _make_bias_tables(height, width, device=None, dtype=None):
        offsets_h = torch.arange(1 - height, height, device=device)
        offsets_w = torch.arange(1 - width, width, device=device)
        offsets = torch.stack(torch.meshgrid(offsets_h, offsets_w, indexing='ij')).flatten(1).transpose(0, 1).float()
        if dtype is not None:
            offsets = offsets.to(dtype)
        coords_h = torch.arange(height, device=device)
        coords_w = torch.arange(width, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij')).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += height - 1
        relative[:, :, 1] += width - 1
        relative[:, :, 0] *= 2 * width - 1
        return offsets, relative.sum(-1)

    def _get_bias_tables(self, height, width, device, dtype):
        key = (int(height), int(width), str(device), str(dtype))
        if key not in self._bias_table_cache:
            self._bias_table_cache[key] = self._make_bias_tables(height, width, device=device, dtype=dtype)
        return self._bias_table_cache[key]

    def forward(self, x, mask=None, hw=None):
        batch, tokens, channels = x.shape
        if hw is not None:
            height, width = map(int, hw)
        elif self.group_size is not None:
            height, width = map(int, self.group_size)
        else:
            height = width = int(round(tokens ** 0.5))
        assert height * width == tokens, f'LRPB: token count {tokens} does not match {height}x{width}'
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)
        if self.position_bias:
            offsets, relative_index = self._get_bias_tables(height, width, x.device, x.dtype)
            bias = self.pos(offsets)[relative_index.reshape(-1)].view(tokens, tokens, -1)
            attention = attention + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            attention = attention.view(batch // windows, windows, self.num_heads, tokens, tokens)
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = self.attn_drop(self.softmax(attention))
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class LRPB_AIFI(nn.Module):
    """AIFI replacement that uses LRPB attention on the deepest feature map."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.lrpb_attention = LRPB_Attention(c1, num_heads=num_heads)
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        _, channels, height, width = src.size()
        attention = self.lrpb_attention(src.flatten(2).permute(0, 2, 1), hw=(height, width))
        attention = attention.permute(0, 2, 1).view(-1, channels, height, width).contiguous()
        src = self.norm1(src + self.dropout1(attention))
        feed_forward = self.fc2(self.dropout(self.act(self.fc1(src))))
        return self.norm2(src + self.dropout2(feed_forward))


class StaticRPEAttention(nn.Module):
    """Static relative-position-bias attention used only by the positional controls."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, ref_size=13):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.ref_size = int(ref_size)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_heads, 2 * self.ref_size - 1, 2 * self.ref_size - 1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self._index_cache = {}
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _relative_position_index(self, height, width):
        key = (int(height), int(width))
        if key not in self._index_cache:
            coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')).flatten(1)
            relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
            relative[:, :, 0] += height - 1
            relative[:, :, 1] += width - 1
            relative[:, :, 0] *= 2 * width - 1
            self._index_cache[key] = relative.sum(-1)
        return self._index_cache[key]

    def _bias(self, height, width, device, dtype):
        table = self.relative_position_bias_table
        if table.shape[1:] != (2 * height - 1, 2 * width - 1):
            table = F.interpolate(table.unsqueeze(0), size=(2 * height - 1, 2 * width - 1), mode='bilinear', align_corners=False).squeeze(0)
        index = self._relative_position_index(height, width).to(device)
        bias = table.reshape(self.num_heads, -1)[:, index.reshape(-1)]
        return bias.reshape(self.num_heads, height * width, height * width).unsqueeze(0).to(dtype=dtype)

    def forward(self, x, hw=None):
        batch, tokens, channels = x.shape
        height, width = map(int, hw) if hw is not None else (int(round(tokens ** 0.5)),) * 2
        assert height * width == tokens, f'StaticRPEAttention: token count {tokens} does not match {height}x{width}'
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1) + self._bias(height, width, x.device, x.dtype)
        x = (self.attn_drop(self.softmax(attention)) @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class PE_AIFI(nn.Module):
    """AIFI positional-encoding controls: sinusoidal, none, or static relative bias."""

    def __init__(self, c1, cm=2048, pos_mode='sinusoidal', num_heads=8, ref_size=13,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        assert pos_mode in ('sinusoidal', 'none', 'static')
        self.pos_mode = pos_mode
        if pos_mode in ('sinusoidal', 'none'):
            self.encoder = TransformerEncoderLayer(c1, cm, num_heads, dropout, act, normalize_before)
        else:
            self.attn = StaticRPEAttention(c1, num_heads=num_heads, proj_drop=dropout, ref_size=ref_size)
            self.fc1 = nn.Conv2d(c1, cm, 1)
            self.fc2 = nn.Conv2d(cm, c1, 1)
            self.norm1 = LayerNorm(c1)
            self.norm2 = LayerNorm(c1)
            self.dropout = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.act = act

    def forward(self, x):
        channels, height, width = x.shape[1:]
        if self.pos_mode in ('sinusoidal', 'none'):
            src = x.flatten(2).permute(0, 2, 1)
            pos = None if self.pos_mode == 'none' else AIFI.build_2d_sincos_position_embedding(width, height, channels).to(x.device, x.dtype)
            return self.encoder(src, pos=pos).permute(0, 2, 1).view(-1, channels, height, width).contiguous()
        attention = self.attn(x.flatten(2).permute(0, 2, 1), hw=(height, width))
        attention = attention.permute(0, 2, 1).view(-1, channels, height, width).contiguous()
        src = self.norm1(x + self.dropout1(attention))
        feed_forward = self.fc2(self.dropout(self.act(self.fc1(src))))
        return self.norm2(src + self.dropout2(feed_forward))


__all__ = ('CAA', 'LRPB_AIFI', 'MCAF', 'MSPC', 'PE_AIFI', 'RepNCSPELAN4')
