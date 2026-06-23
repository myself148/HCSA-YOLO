import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import DFL
from ..modules.conv import autopad

from ultralytics.utils.tal import dist2bbox, make_anchors, dist2rbox

__all__ = ['Detect_STAH', 'OBB_STAH'
           ]

class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.la_conv1 = nn.Conv2d( self.in_channels,  self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d( self.in_channels // la_down_rate,  self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
        self.reduction_conv = Conv_GN(self.in_channels, self.feat_channels, 1)
        self.init_weights()
        
    def init_weights(self):
        # self.la_conv1.weight.normal_(std=0.001)
        # self.la_conv2.weight.normal_(std=0.001)
        # self.la_conv2.bias.data.zero_()
        # self.reduction_conv.conv.weight.normal_(std=0.01)
        
        torch.nn.init.normal_(self.la_conv1.weight.data, mean=0, std=0.001)
        torch.nn.init.normal_(self.la_conv2.weight.data, mean=0, std=0.001)
        torch.nn.init.zeros_(self.la_conv2.bias.data)
        torch.nn.init.normal_(self.reduction_conv.conv.weight.data, mean=0, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs.
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
                          self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        feat = self.reduction_conv.gn(feat)
        feat = self.reduction_conv.act(feat)

        return feat

class Detect_STAH(nn.Module):
    # Task Dynamic Align Detection Head

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv11 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 3)) for x in ch)
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc // 2, 3), Conv_GN(hidc // 2, hidc // 2, 3))
        self.cls_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.reg_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.DyDCNV2 = DyDCNv2(hidc // 2, hidc // 2)
        self.spatial_conv_offset = nn.Conv2d(hidc, 3 * 3 * 3, 3, padding=1)
        self.offset_dim = 2 * 3 * 3
        self.cls_prob_conv1 = nn.Conv2d(hidc, hidc // 4, 1)
        self.cls_prob_conv2 = nn.Conv2d(hidc // 4, 1, 3, padding=1)
        self.cv2 = nn.Conv2d(hidc // 2, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc // 2, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])  # Convert different channel inputs to hidc
            stack_res_list = [self.share_conv[0](x[i])]
            stack_res_list.extend(m(stack_res_list[-1]) for m in self.share_conv[1:])
            feat = torch.cat(stack_res_list, dim=1) # 得到feat
            
            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)
            
            # reg alignment
            offset_and_mask = self.spatial_conv_offset(feat)
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
            reg_feat = self.DyDCNV2(reg_feat, offset, mask)
            
            # cls alignment
            cls_prob = self.cls_prob_conv2(F.relu(self.cls_prob_conv1(feat))).sigmoid()
            
            x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)), self.cv3(cls_feat * cls_prob)), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides



class OBB_STAH(Detect_STAH):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, hidc=256, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, hidc, ch)
        self.ne = ne  # number of extra parameters
        self.detect = Detect_STAH.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv_GN(x, c4, 1), Conv_GN(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes):
        """Decode rotated bounding boxes."""
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides

class DEConv_GN(DEConv):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self, dim):
        super().__init__(dim)
        
        self.bn = nn.GroupNorm(16, dim)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(nn.Conv2d(n, k, 1) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



