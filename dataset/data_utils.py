import numpy as np
import cv2
import torch
import torch.nn.functional as F


def padding(x,y):
    h,w,c = x.shape
    size = max(h,w)
    paddingh = (size-h)//2
    paddingw = (size-w)//2
    temp_x = np.zeros((size,size,c))
    temp_y = np.zeros((size,size))
    temp_x[paddingh:h+paddingh,paddingw:w+paddingw,:] = x
    temp_y[paddingh:h+paddingh,paddingw:w+paddingw] = y
    return temp_x,temp_y

def random_crop(x,y, c=None):
    if c is None:
        h,w = y.shape
        randh = np.random.randint(h/8)
        randw = np.random.randint(w/8)
        randf = np.random.randint(10)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth,h+offseth-randh, offsetw, w+offsetw-randw
        if randf >= 5:
            x = x[::, ::-1, ::]
            y = y[::, ::-1]
        return x[p0:p1,p2:p3],y[p0:p1,p2:p3]
    else:
        h,w = y.shape
        randh = np.random.randint(h/8)
        randw = np.random.randint(w/8)
        randf = np.random.randint(10)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth,h+offseth-randh, offsetw, w+offsetw-randw
        if randf >= 5:
            x = x[::, ::-1, ::]
            y = y[::, ::-1]
            c = c[::, ::-1]
        return x[p0:p1,p2:p3],y[p0:p1,p2:p3], c[p0:p1,p2:p3]

def random_rotate(x,y, c=None):
    if c is None:
        angle = np.random.randint(-25,25)
        h, w = y.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(x, M, (w, h)),cv2.warpAffine(y, M, (w, h))
    else:
        angle = np.random.randint(-25,25)
        h, w = y.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(x, M, (w, h)),cv2.warpAffine(y, M, (w, h)), cv2.warpAffine(c, M, (w, h))

def random_light(x):
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)


def generate_attention_mask(gt_seg, target_size=(80, 80), sigma=8.0, base_value=0.3):
    if isinstance(gt_seg, np.ndarray):
        gt_seg = torch.from_numpy(gt_seg).float()

    if gt_seg.dim() == 2:  # [H, W] -> [1, 1, H, W]
        gt_seg = gt_seg.unsqueeze(0).unsqueeze(0)
    elif gt_seg.dim() == 3:
        gt_seg = gt_seg.unsqueeze(0 if gt_seg.size(0) != 1 else 1)
    elif gt_seg.dim() == 4 and gt_seg.size(1) != 1:  # [B, C, H, W] -> [B, 1, H, W]
        gt_seg = gt_seg[:, :1, :, :]

    B, C, H, W = gt_seg.size()  # [B, 1, 640, 640]

    attention_mask = torch.ones_like(gt_seg) * base_value  # [B, 1, 640, 640]

    for b in range(B):
        gt = gt_seg[b, 0]
        target_coords = torch.nonzero(gt == 1)  # [N, 2]
        if target_coords.size(0) == 0:
            continue

        center_y, center_x = target_coords.float().mean(dim=0)  # [2]

        y, x = torch.meshgrid(
            torch.arange(H, device=gt.device),
            torch.arange(W, device=gt.device),
            indexing='ij',
        )
        dist = (x - center_x) ** 2 + (y - center_y) ** 2
        gauss = torch.exp(-dist / (2 * sigma ** 2))  # [640, 640]
        gauss = gauss.clamp(max=1.0)

        attention_mask[b, 0] = torch.max(attention_mask[b, 0], gauss)

    attention_mask = F.interpolate(attention_mask, size=target_size, mode='bilinear', align_corners=False)
    attention_mask = attention_mask.clamp(min=base_value, max=1.0)
    attention_mask = attention_mask.squeeze(0)  # [B, 1, 80, 80] -> [1, 80, 80] if B=1
    return attention_mask


def generate_noisy_attention_mask(gt_attention_mask, noise_level=0.5, max_false_targets=2, sigma=1.0, base_value=0.3):
    B, C, H, W = gt_attention_mask.size()  # [B, 1, H, W]
    noisy_mask = gt_attention_mask.clone()  # [B, 1, H, W]


    noise = torch.randn_like(noisy_mask) * noise_level
    noisy_mask = noisy_mask + noise
    noisy_mask = noisy_mask.clamp(min=base_value, max=1.0)

    device = gt_attention_mask.device
    for b in range(B):
        num_false_targets = torch.randint(0, max_false_targets + 1, (1,), device=device).item()

        if num_false_targets == 0:
            continue

        for _ in range(num_false_targets):
            center_y = torch.randint(5, H - 5, (1,), device=device).item()
            center_x = torch.randint(5, W - 5, (1,), device=device).item()

            y, x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij',
            )
            dist = (x - center_x) ** 2 + (y - center_y) ** 2
            gauss = torch.exp(-dist / (2 * sigma ** 2))  # [H, W]
            gauss = gauss.clamp(max=1.0)

            noisy_mask[b, 0] = torch.max(noisy_mask[b, 0], gauss)

    noisy_mask = noisy_mask.clamp(min=base_value, max=1.0)

    return noisy_mask  # [B, 1, H, W]
