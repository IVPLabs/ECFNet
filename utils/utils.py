import matplotlib.pyplot as plt
import openpyxl
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import os
from torchvision.transforms import ToPILImage



def create_Excel(file_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Training Metrics"
    sheet.append([
        "Epoch", "Train Loss", "Test Loss", "TP", "FN", "FP", "TN", "MAE", "Max F-score", "IoU", "TPR"
    ])
    return workbook, sheet




def visualize_feature_maps(feature_maps, original_image, save_dir):
    original_image = original_image.astype(np.uint8)

    for i, feature_map in enumerate(feature_maps):
        upsampled_feature_map = F.interpolate(feature_map, size=(original_image.shape[0], original_image.shape[1]), mode='bilinear', align_corners=False)
        upsampled_feature_map = upsampled_feature_map[0].cpu().numpy().mean(axis=0)

        normalized_map = (upsampled_feature_map - upsampled_feature_map.min()) / (upsampled_feature_map.max() - upsampled_feature_map.min() + 1e-8)

        heatmap = cv2.applyColorMap((normalized_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

        if original_image.ndim == 2 or original_image.shape[-1] == 1:
            original_image_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            original_image_colored = original_image

        overlay = cv2.addWeighted(original_image_colored, 0.6, heatmap, 0.4, 0)

        save_path = os.path.join(save_dir, f"stage_{i}_heat_map.png")
        cv2.imwrite(save_path, overlay)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        #sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

