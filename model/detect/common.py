import torch
import torch.nn as nn
from model.blocks import C3k2
import os
import numpy as np

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, d=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x, act=True):
        return self.act(self.conv(x)) if act else self.conv(x)

class ExpSlicer(nn.Module):
    def __init__(self, c, ratio=8, threshold=0.5, mask_only=False,
                 cluster_only=False, train_target_num=6):
        super().__init__()
        self.c = c
        self.ratio = ratio
        self.threshold = threshold
        self.mask_only = mask_only
        self.cluster_only = cluster_only

        self.grid = None
        self.train_target_num = train_target_num

    def forward(self, x):
        x, mask_pred = x
        bs, c, ny, nx = x.shape
        device, dtype = x.device, x.dtype
        assert c == self.c, f'{c} - {self.c}'

        if getattr(self, 'mask_only', False):
            return x, self.threshold


        if self.training:
            return self.uni_slicer(
                x, mask_pred, target_num=self.train_target_num, device=device
            )
        else:
            patches = []
            offsets = []
            block_h, block_w = ny // 20, nx // 20
            patch_h, patch_w = block_h * 3, block_w * 3
            mask_pred = mask_pred.squeeze(1)
            for bi in range(bs):
                mask = mask_pred[bi]  # [20, 20]
                covered_mask = torch.zeros_like(mask, dtype=torch.bool, device=device)

                while True:
                    valid_indices = (mask == 1) & (~covered_mask)
                    valid_indices = valid_indices.nonzero(as_tuple=False)  # [N_valid, 2]

                    if len(valid_indices) == 0:
                        break

                    idx = valid_indices[0]
                    gy, gx = idx[0].item(), idx[1].item()
                    center_y, center_x = gy * block_h, gx * block_w

                    if center_y - block_h < 0:
                        y1 = 0
                        y2 = patch_h
                    elif center_y + block_h * 2 > ny:
                        y2 = ny
                        y1 = ny - patch_h
                    else:
                        y1 = center_y - block_h
                        y2 = center_y + block_h * 2

                    if center_x - block_w < 0:
                        x1 = 0
                        x2 = patch_w
                    elif center_x + block_w * 2 > nx:
                        x2 = nx
                        x1 = nx - patch_w
                    else:
                        x1 = center_x - block_w
                        x2 = center_x + block_w * 2

                    assert y2 - y1 == patch_h and x2 - x1 == patch_w, f"Patch size error: {(y2 - y1)}x{(x2 - x1)}"

                    patch = x[bi, :, y1:y2, x1:x2]  # [c, 24, 18]
                    patches.append(patch)
                    offsets.append(torch.tensor([bi, x1, y1, x2, y2], dtype=torch.float, device=device))

                    gy1 = max(0, y1 // block_h)
                    gy2 = min(20, (y2 + block_h - 1) // block_h)
                    gx1 = max(0, x1 // block_w)
                    gx2 = min(20, (x2 + block_w - 1) // block_w)
                    covered_mask[gy1:gy2, gx1:gx2] = True


            if patches:
                patches = torch.stack(patches, dim=0)  # [N, c, 24, 18]
                offsets = torch.stack(offsets, dim=0)  # [N, 5]
            else:
                patches = torch.zeros((0, c, patch_h, patch_w), device=device)
                offsets = torch.zeros((0, 5), device=device)

            return patches, offsets


    @staticmethod
    def get_offsets_by_clusters(total_clusters):
        offsets = []
        for bi, clusters in enumerate(total_clusters):
            b = torch.full_like(clusters[:, :1], bi)
            offsets.append(torch.cat((b, clusters), dim=1))
        return torch.cat(offsets)


    def uni_slicer(self, feat, mask_pred, target_num=1, device=None):
        B, C, H, W = feat.shape
        _, _, mask_h, mask_w = mask_pred.shape
        block_h, block_w = H // mask_h, W // mask_w
        assert H % mask_h == 0 and W % mask_w == 0, f"Feature map size {H}x{W} must be divisible by mask size {mask_h}x{mask_w}"

        mask_pred = mask_pred.squeeze(1)
        patches, grid_off = [], []

        def find_parent(parent, i):
            if parent[i] != i:
                parent[i] = find_parent(parent, parent[i])
            return parent[i]

        def union(parent, rank, i, j):
            pi, pj = find_parent(parent, i), find_parent(parent, j)
            if pi != pj:
                if rank[pi] < rank[pj]:
                    pi, pj = pj, pi
                parent[pj] = pi
                if rank[pi] == rank[pj]:
                    rank[pi] += 1

        for b in range(B):
            mask = mask_pred[b]
            valid_indices = (mask == 1).nonzero(as_tuple=False)  # [num_points, 2]
            num_points = len(valid_indices)

            if num_points == 0:
                selected_indices = torch.randperm(mask_h * mask_w, device=mask_pred.device)[:target_num]
                selected_indices = torch.stack((selected_indices // mask_w, selected_indices % mask_w), dim=1)
            else:
                parent = list(range(num_points))
                rank = [0] * num_points
                for i in range(num_points):
                    for j in range(i + 1, num_points):
                        y1, x1 = valid_indices[i]
                        y2, x2 = valid_indices[j]
                        if abs(y1 - y2) + abs(x1 - x2) <= 1:
                            union(parent, rank, i, j)

                groups = {}
                for i in range(num_points):
                    root = find_parent(parent, i)
                    if root not in groups:
                        groups[root] = i
                num_valid = len(groups)

                group_indices = list(groups.values())
                if num_valid < target_num:
                    remaining = target_num - num_valid
                    used_indices = (
                        valid_indices[:, 0] * mask_w + valid_indices[:, 1]
                    ).to(device)
                    all_indices = torch.arange(mask_h * mask_w, device=device)
                    available_indices = all_indices[~torch.isin(all_indices, used_indices)]
                    if len(available_indices) < remaining:
                        raise ValueError(f"Not enough unique indices for {remaining} additional patches")
                    random_indices = available_indices[torch.randperm(len(available_indices))[:remaining]]
                    random_coords = torch.stack((random_indices // mask_w, random_indices % mask_w), dim=1)
                    selected_indices = torch.cat([valid_indices[group_indices], random_coords], dim=0)
                elif num_valid > target_num:
                    perm = torch.randperm(num_valid)[:target_num]
                    selected_indices = valid_indices[[group_indices[i] for i in perm]]
                else:
                    selected_indices = valid_indices[group_indices]
            for idx in selected_indices:
                gy, gx = idx[0].item(), idx[1].item()
                center_y, center_x = gy * block_h, gx * block_w
                patch_h, patch_w = block_h * 3, block_w * 3

                y1 = max(0, center_y - block_h)
                y2 = min(H, center_y + block_h * 2)
                if y2 - y1 != patch_h:
                    y2 = min(H, y1 + patch_h)
                    y1 = max(0, y2 - patch_h)

                x1 = max(0, center_x - block_w)
                x2 = min(W, center_x + block_w * 2)
                if x2 - x1 != patch_w:
                    x2 = min(W, x1 + patch_w)
                    x1 = max(0, x2 - patch_w)

                assert y2 - y1 == patch_h and x2 - x1 == patch_w
                patches.append(feat[b, :, y1:y2, x1:x2])
                grid_off.append(torch.tensor([b, x1, y1, x2, y2], dtype=torch.float, device=device))

        if patches:
            patches = torch.stack(patches, dim=0)
            grid_off = torch.stack(grid_off, dim=0)
            assert len(patches) == B * target_num, f"Expected {B * target_num} patches, got {len(patches)}"
        else:
            patches = torch.zeros(1, C, block_h * 3, block_w * 3, device=device)
            grid_off = torch.zeros(1, 5, device=device)

        return patches, grid_off



class YOLOXHead(nn.Module):
    # https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
    def __init__(self, c1, nc, na, w=1.0):
        super(YOLOXHead, self).__init__()
        print("Use YOLOXHead")

        self.nc = nc
        self.na = na
        c = int(256 * w)
        self.stem = Conv(c1, c, 1)
        self.cls_conv = nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1), Conv(c, c, 3, 1))
        self.reg_conv = nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1), Conv(c, c, 3, 1))


        self.cls_pred = nn.Conv2d(c, nc * na, 1)
        self.reg_pred = nn.Conv2d(c, 4 * na, 1)
        self.obj_pred = nn.Conv2d(c, 1 * na, 1)

    def forward(self, x):
        bs, _, ny, nx = x.shape
        stem = self.stem(x)
        cls_feat = self.cls_conv(stem)
        reg_feat = self.reg_conv(stem)

        cls = self.cls_pred(cls_feat).view(bs, self.na, self.nc, ny, nx)  # 16,3,1,24,24
        reg = self.reg_pred(reg_feat).view(bs, self.na, 4, ny, nx)  # 16,3,4,24,24
        obj = self.obj_pred(reg_feat).view(bs, self.na, 1, ny, nx)  # 16,3,1,24,24

        y = torch.cat((reg, obj, cls), 2)
        return y.view(bs, -1, ny, nx)

class YOLOv6Head(YOLOXHead):
    # https://github.com/meituan/YOLOv6/blob/main/yolov6/models/effidehead.py
    def __init__(self, c1, nc, na):
        super(YOLOv6Head, self).__init__(c1, nc, na)
        print("Use YOLOv6Head")
        self.nc = nc
        self.na = na
        c = c1
        self.stem = Conv(c1, c, 1)
        self.cls_conv = Conv(c, c, 3, 1)
        self.reg_conv = Conv(c, c, 3, 1)

        self.cls_pred = nn.Conv2d(c, nc * na, 1)
        self.reg_pred = nn.Conv2d(c, 4 * na, 1)
        self.obj_pred = nn.Conv2d(c, 1 * na, 1)

class YOLO11Head(YOLOXHead):
    def __init__(self, c1, nc, na):
        super(YOLO11Head, self).__init__(c1, nc, na)
        print("Use YOLO11Head")

        self.nc = nc
        self.na = na
        c = c1
        self.stem = C3k2(c1=c, c2=c, n=1, shortcut=True, g=1, e=0.25)
        self.cls_conv = nn.Sequential(
            C3k2(c1=c, c2=c, n=1, shortcut=True, g=1, e=0.25),
            C3k2(c1=c, c2=c, n=1, shortcut=True, g=1, e=0.25),
            C3k2(c1=c, c2=c, n=1, shortcut=True, g=1, e=0.25)
        )
        self.reg_conv = nn.Sequential(
            C3k2(c1=c, c2=c, n=1, shortcut=True, g=1, e=0.25),
            C3k2(c1=c, c2=c, n=1, shortcut=True, g=1, e=0.25),
            C3k2(c1=c, c2=c, n=1, shortcut=True, g=1, e=0.25)
        )

        self.cls_pred = nn.Conv2d(c, nc * na, 1)
        self.reg_pred = nn.Conv2d(c, 4 * na, 1)
        self.obj_pred = nn.Conv2d(c, 1 * na, 1)



def get_decoupled_heads(ch, nc, na, type='YOLOv6Head'):
    return nn.ModuleList(eval(type)(x, nc, na) for x in ch)

def extract_features(features_list, offsets, ratios, save_dir=None):

    b_ids = offsets[:, 0].long()
    device = offsets.device
    patches_list = []
    patch_coords_list = [] # for visual

    for idx, (feature, ratio) in enumerate(zip(features_list, ratios)):
        x1s = (offsets[:, 1] / ratio).long()
        y1s = (offsets[:, 2] / ratio).long()
        x2s = (offsets[:, 3] / ratio).long()
        y2s = (offsets[:, 4] / ratio).long()

        h = (y2s - y1s)[0].item()
        w = (x2s - x1s)[0].item()
        N, C = offsets.size(0), feature.size(1)
        patches = torch.empty((N, C, h, w), device=device)
        patch_coords = torch.stack([x1s, y1s, x2s, y2s], dim=-1)

        for i in range(N):
            patches[i] = feature[b_ids[i], :, y1s[i]:y2s[i], x1s[i]:x2s[i]]

        patches_list.append(patches)
        patch_coords_list.append(patch_coords)


    return patches_list  # List of [N, C, h, w]





