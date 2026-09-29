"""HCSA-YOLO custom modules collected in one file.

This file contains the five modules used by HCSA-YOLO:

* DSWI: the modified SNI upsampling module.
* ACCPM: attention-based cross-stage partial multi-scale perception module
  (named ``ACCPO`` in the original training code).
* SPDConv: space-to-depth convolution.
* ERLConv: the modified GSConv module.
* STAH: the detection head derived from TADDH
  (named ``Detect_STAH`` in the original training code).

The feature modules depend only on PyTorch. STAH additionally requires MMCV and
MMEngine because its regression branch uses modulated deformable convolution.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import EMA

try:
    from mmcv.cnn import build_norm_layer
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
except ImportError:  # Allow the four feature modules to be imported without MMCV.
    build_norm_layer = None
    ModulatedDeformConv2d = None


__all__ = [
    "DSWI",
    "ACCPM",
    "ACCPO",
    "SPDConv",
    "ERLConv",
    "STAH",
    "Detect_STAH",
    "OBB_STAH",
]


def autopad(k, p=None, d=1):
    """Return padding that preserves the spatial size for stride 1."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * x - d + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Convolution followed by batch normalization and an activation."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FGM(nn.Module):
    """Frequency-domain gated modulation used in ACCPM."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2_fft = torch.fft.fft2(self.dwconv2(x), norm="backward")
        out = torch.fft.ifft2(x1 * x2_fft, dim=(-2, -1), norm="backward").abs()
        return out * self.alpha + x * self.beta


class OD_Attention(nn.Module):
    """Omni-dimensional attention for ODConv."""

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=1):
        super().__init__()
        attention_channels = max(int(in_planes * reduction), 16)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc = nn.Conv2d(attention_channels, in_planes, 1)

        self.filter_fc = None
        if not (in_planes == groups and in_planes == out_planes):
            self.filter_fc = nn.Conv2d(attention_channels, out_planes, 1)
        self.spatial_fc = None if kernel_size == 1 else nn.Conv2d(attention_channels, kernel_size**2, 1)
        self.kernel_fc = None if kernel_num == 1 else nn.Conv2d(attention_channels, kernel_num, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.fc(self.avgpool(x))
        # Ultralytics initializes detection strides with a single image. Since
        # global pooling produces a 1x1 tensor, regular training-mode BatchNorm
        # has only one value per channel and would otherwise raise an error.
        if self.training and x.numel() // x.shape[1] == 1:
            x = F.batch_norm(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                momentum=self.bn.momentum,
                eps=self.bn.eps,
            )
        else:
            x = self.bn(x)
        x = self.relu(x)
        b = x.shape[0]
        channel = torch.sigmoid(self.channel_fc(x).view(b, -1, 1, 1) / self.temperature)
        filt = 1.0 if self.filter_fc is None else torch.sigmoid(self.filter_fc(x).view(b, -1, 1, 1) / self.temperature)
        spatial = 1.0
        if self.spatial_fc is not None:
            spatial = torch.sigmoid(
                self.spatial_fc(x).view(b, 1, 1, 1, self.kernel_size, self.kernel_size) / self.temperature
            )
        kernel = 1.0
        if self.kernel_fc is not None:
            kernel = F.softmax(self.kernel_fc(x).view(b, -1, 1, 1, 1, 1) / self.temperature, dim=1)
        return channel, filt, spatial, kernel


class ODConv2d(nn.Module):
    """Omni-dimensional dynamic convolution used by ACCPM."""

    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=None, dilation=1,
        groups=1, reduction=0.0625, kernel_num=1,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = autopad(kernel_size, padding, dilation)
        self.dilation = dilation
        self.groups = groups
        self.attention = OD_Attention(
            in_planes, out_planes, kernel_size, groups, reduction, kernel_num
        )
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size)
        )
        for weight in self.weight:
            nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        channel, filt, spatial, kernel = self.attention(x)
        b, _, h, w = x.shape
        x = (x * channel).reshape(1, -1, h, w)
        weight = (spatial * kernel * self.weight.unsqueeze(0)).sum(dim=1)
        weight = weight.view(-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size)
        out = F.conv2d(
            x, weight, stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups * b,
        )
        out = out.view(b, self.out_planes, out.shape[-2], out.shape[-1])
        return out * filt


class OmniKernelImproved(nn.Module):
    """Large-kernel spatial/frequency branch used by ACCPM."""

    def __init__(self, dim):
        super().__init__()
        kernel = 31
        pad = kernel // 2
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, 1)
        self.dw_13 = nn.Conv2d(dim, dim, (1, kernel), padding=(0, pad), groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, (kernel, 1), padding=(pad, 0), groups=dim)
        self.dw_33 = ODConv2d(dim, dim, kernel, padding=pad, groups=dim, reduction=0.0625)
        self.dw_11 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.act = nn.SiLU()
        self.ema_attention = EMA(dim)
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)
        attended = self.fgm(self.ema_attention(out))
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + attended
        return self.out_conv(self.act(out))


class ACCPM(nn.Module):
    """Attention-based cross-stage partial multi-scale perception module."""

    def __init__(self, dim, e=0.5):
        super().__init__()
        hidden = int(dim * e)
        if hidden <= 0 or hidden >= dim:
            raise ValueError(f"ACCPM requires 0 < int(dim * e) < dim, got dim={dim}, e={e}")
        self.hidden = hidden
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.perception = OmniKernelImproved(hidden)
        self.fusion_conv = Conv(dim, dim, 1)
        self.attention = EMA(dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        perception, identity = torch.split(self.cv1(x), [self.hidden, x.shape[1] - self.hidden], dim=1)
        fused = torch.cat((self.perception(perception), identity), dim=1)
        fused = self.attention(self.fusion_conv(fused))
        return self.norm(self.cv2(fused))


# Compatibility name used in the original training project.
ACCPO = ACCPM


class DSWI(nn.Module):
    """Dimension-preserving scaled weighted interpolation (modified SNI)."""

    def __init__(self, up_f=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=up_f, mode="nearest")
        self.alpha = nn.Parameter(torch.tensor(1.0 / up_f**2, dtype=torch.float32))

    def forward(self, x):
        return self.alpha * self.upsample(x)


class SPDConv(nn.Module):
    """Space-to-depth convolution."""

    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension  # Kept for compatibility with the original implementation.
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(f"SPDConv requires even H and W, got {tuple(x.shape[-2:])}")
        x = torch.cat(
            (x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1
        )
        return self.conv(x)


class ERLConv(nn.Module):
    """Enhanced receptive-field lightweight convolution (modified GSConv)."""

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True, k_aux=5):
        super().__init__()
        if c2 % 2:
            raise ValueError(f"ERLConv requires an even output-channel count, got c2={c2}")
        hidden = c2 // 2
        self.cv1 = Conv(c1, hidden, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, k_aux, padding=(k_aux - 1) // 2, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.cv1(x)
        y = torch.cat((x1, self.cv2(x1)), dim=1)
        b, c, h, w = y.shape
        return y.reshape(b, 2, c // 2, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)


class DFL(nn.Module):
    """Integral layer for Distribution Focal Loss predictions."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x):
        b, _, anchors = x.shape
        return self.conv(x.view(b, 4, self.c1, anchors).transpose(2, 1).softmax(1)).view(b, 4, anchors)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor centers and their stride tensor."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform left/top/right/bottom distances to bounding boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1, x2y2 = anchor_points - lt, anchor_points + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """Transform distances and angles to rotated bounding boxes."""
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    xy = torch.cat((xf * cos - yf * sin, xf * sin + yf * cos), dim=dim) + anchor_points
    return torch.cat((xy, lt + rb), dim=dim)


class Conv_GN(nn.Module):
    """Convolution followed by group normalization and SiLU."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if c2 % 16:
            raise ValueError(f"Conv_GN requires c2 divisible by 16, got c2={c2}")
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class TaskDecomposition(nn.Module):
    """Layer-attention task decomposition used by STAH."""

    def __init__(self, feat_channels, stacked_convs, la_down_rate=8):
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = feat_channels * stacked_convs
        self.la_conv1 = nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d(self.in_channels // la_down_rate, stacked_convs, 1)
        self.sigmoid = nn.Sigmoid()
        self.reduction_conv = Conv_GN(self.in_channels, feat_channels, 1)
        nn.init.normal_(self.la_conv1.weight, mean=0, std=0.001)
        nn.init.normal_(self.la_conv2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.la_conv2.bias)
        nn.init.normal_(self.reduction_conv.conv.weight, mean=0, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, _, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, 1)
        weight = self.sigmoid(self.la_conv2(self.relu(self.la_conv1(avg_feat))))
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * self.reduction_conv.conv.weight.reshape(
            1, self.feat_channels, self.stacked_convs, self.feat_channels
        )
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = torch.bmm(conv_weight, feat.reshape(b, self.in_channels, h * w)).reshape(
            b, self.feat_channels, h, w
        )
        return self.reduction_conv.act(self.reduction_conv.gn(feat))


class DyDCNv2(nn.Module):
    """Modulated deformable convolution used for STAH regression alignment."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if ModulatedDeformConv2d is None or build_norm_layer is None:
            raise ImportError("STAH requires mmcv and mmengine with the mmcv.ops extensions installed")
        self.conv = ModulatedDeformConv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = build_norm_layer(dict(type="GN", num_groups=16, requires_grad=True), out_channels)[1]

    def forward(self, x, offset, mask):
        return self.norm(self.conv(x.contiguous(), offset, mask))


class STAH(nn.Module):
    """Spatial-Task Alignment Head derived from TADDH."""

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.conv = nn.ModuleList(Conv_GN(c, hidc, 3) for c in ch)
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc // 2, 3), Conv_GN(hidc // 2, hidc // 2, 3))
        self.cls_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.reg_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.dcn = DyDCNv2(hidc // 2, hidc // 2)
        self.spatial_conv_offset = nn.Conv2d(hidc, 27, 3, padding=1)
        self.offset_dim = 18
        self.cls_prob_conv1 = nn.Conv2d(hidc, hidc // 4, 1)
        self.cls_prob_conv2 = nn.Conv2d(hidc // 4, 1, 3, padding=1)
        self.cv2 = nn.Conv2d(hidc // 2, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc // 2, nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            first = self.share_conv[0](x[i])
            second = self.share_conv[1](first)
            feat = torch.cat((first, second), dim=1)
            avg_feat = F.adaptive_avg_pool2d(feat, 1)
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)
            offset_and_mask = self.spatial_conv_offset(feat)
            offset = offset_and_mask[:, : self.offset_dim]
            mask = offset_and_mask[:, self.offset_dim :].sigmoid()
            reg_feat = self.dcn(reg_feat, offset, mask)
            cls_prob = self.cls_prob_conv2(F.relu(self.cls_prob_conv1(feat))).sigmoid()
            x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)), self.cv3(cls_feat * cls_prob)), dim=1)

        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (value.transpose(0, 1) for value in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)
        dbox = self.decode_bboxes(box)
        y = torch.cat((dbox, cls.sigmoid()), dim=1)
        return y if self.export else (y, x)

    def bias_init(self):
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


# Compatibility name used by the original model YAML.
Detect_STAH = STAH


class OBB_STAH(STAH):
    """Oriented-bounding-box variant of STAH."""

    def __init__(self, nc=80, ne=1, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.ne = ne
        self.detect = STAH.forward
        c4 = max(ch[0] // 4, ne)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv_GN(c, c4, 1), Conv_GN(c4, c4, 3), nn.Conv2d(c4, ne, 1)) for c in ch
        )

    def forward(self, x):
        batch = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(batch, self.ne, -1) for i in range(self.nl)], dim=2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        predictions = self.detect(self, x)
        if self.training:
            return predictions, angle
        if self.export:
            return torch.cat((predictions, angle), dim=1)
        return torch.cat((predictions[0], angle), dim=1), (predictions[1], angle)

    def decode_bboxes(self, bboxes):
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides
