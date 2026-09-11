from .common import *
import torch.nn.functional as F

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

        self.m = get_decoupled_heads(ch, self.nc, self.na, type="YOLOXHead")  # decoupled head
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.stride = torch.tensor([4., 8.])
        self.sparse = False
        self.register_buffer('sparse_gird', torch.zeros(1))


    def forward(self, x):

        masks, offsets, indices_per_layer = None, None, None

        if isinstance(x, tuple):
            if len(x) == 2:
                x, offsets = x  # offsets(bi,x1,y1,x2,y2)
            else:
                x, offsets, masks = x
                if offsets is not None and hasattr(self, 'sparse') and self.sparse:
                    indices_per_layer = self.get_indices(offsets, masks[0])
            if offsets is not None:
                img_bs = torch.max(offsets[:, 0]).int().item() + 1
            else:
                img_bs = x[0].shape[0]
        else:
            img_bs = x[0].shape[0]

        device = x[0].device
        z = []  # inference output
        patch_offsets = []

        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            if offsets is not None:
                r = (2 ** (i - 1)) if self.nl == 4 else 2 ** i
                patch_off = torch.cat((offsets[:, :1], offsets[:, 1:] / r), dim=1)  # TODO: from 4 to 32
                patch_off_xy = patch_off[:, 1:3].view(-1, 1, 1, 1, 2)
                patch_offsets.append(patch_off)

            if indices_per_layer is not None:
                sp_x = self.m[i](x[i], indices_per_layer[i])  # sparse conv
            else:
                sp_x = None
                x[i] = self.m[i](x[i])  # conv
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # 16,3,24,24,6

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != (ny, nx) or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(device)
                if sp_x is not None:
                    y = sp_x.features.sigmoid().view(-1, self.na, self.no)
                    bi, yi, xi = sp_x.indices.long().T
                    assert offsets is not None
                    grid_off = self.grid[i][0, 0, yi, xi].view(-1, 1, 2) + patch_off_xy[bi, ...].view(-1, 1, 2)
                    anch_wh = self.anchor_grid[i].view(1, self.na, 2)
                    batch_ind = offsets[bi, 0]  # [num_patches, 5] --> [num_objects, 5], compatible for box concat
                else:
                    y = x[i].sigmoid()
                    anch_wh = self.anchor_grid[i].view(1, self.na, 1, 1, 2)
                    if offsets is not None:
                        grid_off = self.grid[i] + patch_off_xy
                        batch_ind = offsets[:, 0]
                    else:
                        grid_off = self.grid[i]
                        batch_ind = None

                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_off) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anch_wh

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + grid_off) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * anch_wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)


                if offsets is not None:
                    pbox = []
                    for bi in range(img_bs):
                        pbox_bi = y[batch_ind == bi]
                        np = len(pbox_bi)
                        if np:
                            pbox.append(pbox_bi.view(-1, self.no))
                        else:
                            pbox.append(torch.zeros((0, self.no), device=device))
                    max_pnum = max([len(boxes) for boxes in pbox])
                    z.append(torch.stack(
                        [torch.cat((boxes, torch.zeros((max_pnum - len(boxes), self.no), device=device))) for boxes in
                         pbox]
                    ))
                else:
                    z.append(y.view(bs, -1, self.no))
                if i == 0:
                    a, gj, gi = 1, 15, 15
                    idx = (a * ny * nx) + (gj * nx) + gi
                else:
                    a, gj, gi = 1, 7, 7
                    idx = (a * ny * nx) + (gj * nx) + gi

        if offsets is not None:
            x = (x, patch_offsets)
        else:
            x = (x, None)

        return x if self.training else (torch.cat(z, 1), x)


    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()




