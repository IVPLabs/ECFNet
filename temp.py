if self.if_denoise:
    noise_attention_mask_80 = generate_noisy_attention_mask(gt_attention_mask[0], sigma=1)
    noise_attention_mask_40 = generate_noisy_attention_mask(gt_attention_mask[1], sigma=0.5)
    noise_attention_mask_20 = generate_noisy_attention_mask(gt_attention_mask[2], sigma=0.25)
    feature_noise_80 = torch.cat((feature_80, noise_attention_mask_80), dim=1)
    feature_noise_40 = torch.cat((feature_40, noise_attention_mask_40), dim=1)
    feature_noise_20 = torch.cat((feature_20, noise_attention_mask_20), dim=1)
    noise_mask_80 = self.denoise_conv_80(self.denoise_conv_80_press(feature_noise_80))
    noise_mask_40 = self.denoise_conv_40(self.denoise_conv_40_press2(self.denoise_conv_40_press1(feature_noise_40)))
    noise_mask_20 = self.denoise_conv_20(
        self.denoise_conv_20_press3(self.denoise_conv_20_press2(self.denoise_conv_20_press1(feature_noise_20))))
    noise_mask = [noise_mask_80, noise_mask_40, noise_mask_20]


gt_attention_mask = [attention_mask.to(device) for attention_mask in gt_attention_mask]
ptr, noise_ptr = model(xtr, gt_attention_mask)
noise_ltr1 = bce_iou_loss(noise_ptr[0], atr_80, pos_weight=1.0)
noise_ltr2 = bce_iou_loss(noise_ptr[1], atr_40, pos_weight=1.0)
noise_ltr3 = bce_iou_loss(noise_ptr[2], atr, pos_weight=1.0)
noise_ltr_all = noise_ltr1 + noise_ltr2 + noise_ltr3


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

        y, x = torch.meshgrid(torch.arange(640, device=gt.device),
                              torch.arange(640, device=gt.device))
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

            y, x = torch.meshgrid(torch.arange(H, device=device),
                                  torch.arange(W, device=device))
            dist = (x - center_x) ** 2 + (y - center_y) ** 2
            gauss = torch.exp(-dist / (2 * sigma ** 2))  # [H, W]
            gauss = gauss.clamp(max=1.0)

            noisy_mask[b, 0] = torch.max(noisy_mask[b, 0], gauss)

    noisy_mask = noisy_mask.clamp(min=base_value, max=1.0)

    return noisy_mask  # [B, 1, H, W]
