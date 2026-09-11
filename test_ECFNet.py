# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from model.detect.utils.general import box_iou, non_max_suppression, scale_coords, xywh2xyxy
from model.detect.utils.metrics import ap_per_class
from model.detect.utils.plots import  LatencyBucket



def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    """。
    Args:
        pred_boxes (Tensor): pred，[N_pred, 4]
        gt_boxes (Tensor): GT ， [N_gt, 4]

    Returns:
        correct (Tensor): [N_pred, 1]
    """
    device = pred_boxes.device
    correct = torch.zeros(pred_boxes.shape[0], 1, dtype=torch.bool, device=device)

    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return correct

    ious = box_iou(pred_boxes, gt_boxes)  # [N_pred, N_gt]
    gt_matched = set()

    for pi in range(ious.shape[0]):
        iou, gi = ious[pi].max(0)
        if iou > iou_threshold and gi.item() not in gt_matched:
            correct[pi] = True
            gt_matched.add(gi.item())

    return correct




def save_yolo_predictions(pred, img_shape, img_name, save_dir="./save_labels"):
    if isinstance(img_name, (tuple, list)):
        if len(img_name) == 1:
            img_name = img_name[0]
        else:
            raise ValueError(f"img_name should be a string or a single-element tuple, got {img_name}")
    elif not isinstance(img_name, str):
        raise ValueError(f"img_name should be a string, got {type(img_name)}: {img_name}")


    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{img_name}.txt")
    print(f"Saving predictions to {save_path}")


    if pred is None or len(pred) == 0:
        print(f"No predictions for {img_name}, skipping...")
        return

    # [x_center, y_center, width, height]
    pred_yolo = torch.zeros_like(pred[:, :4])  # [n, 4]
    pred_yolo[:, 0] = (pred[:, 0] + pred[:, 2]) / 2  # x_center
    pred_yolo[:, 1] = (pred[:, 1] + pred[:, 3]) / 2  # y_center
    pred_yolo[:, 2] = pred[:, 2] - pred[:, 0]  # width
    pred_yolo[:, 3] = pred[:, 3] - pred[:, 1]  # height

    #  [0, 1]
    img_h, img_w = img_shape
    pred_yolo[:, [0, 2]] /= img_w  # x_center, width
    pred_yolo[:, [1, 3]] /= img_h  # y_center, height


    with open(save_path, "w") as f:
        for i, p in enumerate(pred):
            cls = int(p[5])
            conf = p[4]
            # class x_center y_center width height conf
            f.write(f"{cls} {pred_yolo[i, 0]:.6f} {pred_yolo[i, 1]:.6f} "
                    f"{pred_yolo[i, 2]:.6f} {pred_yolo[i, 3]:.6f} {conf:.6f}\n")


@torch.no_grad()
def test(model, dataloader, compute_loss, device, conf_thres=0.5, iou_thres=0.7, epoch=0, num_classes=1, width=640, height=640, save_labels=False, max_batches=0):
    print("testing ECFNet...")
    model.eval()
    total_loss = 0.0
    losses = torch.zeros(3, device=device)  # box, obj, cls
    stats = []
    seen = 0


    with torch.no_grad():
        for batch_i, (img, mask, targets, shapes, name) in enumerate(dataloader):
            if max_batches and batch_i >= max_batches:
                break
            img = img.to(device).float()
            targets = targets.to(device)
            targets = targets.squeeze(0)

            (out, p_det), pred_masks = model(img)

            x, offsets = p_det
            if offsets is not None:
                loss, loss_items = compute_loss(p_det, targets, imgsz=img.shape)

                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)

                out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)

                for si, pred in enumerate(out):
                    img_name = name

                    if save_labels:
                        save_yolo_predictions(pred, [640, 640], img_name)

                    labels = targets[targets[:, 0] == si, 1:]  #[cls, x, y, w, h]
                    tbox = xywh2xyxy(labels[:, 1:5])  # [x1, y1, x2, y2]
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []
                    seen += 1
                    if len(pred) == 0:
                        if nl:
                            stats.append((torch.zeros(0, 1, dtype=torch.bool, device="cpu"),
                                          torch.Tensor(), torch.Tensor(), tcls))
                        continue

                    predn = pred.clone()
                    scale_coords(img[si].shape[1:], predn[:, :4], [512, 640], ((1.25, 1.0), (0, 0)))

                    correct = torch.zeros(pred.shape[0], 1, dtype=torch.bool, device=device)  # [n_p, 1]
                    if nl:
                        tbox = xywh2xyxy(labels[:, 1:5])  # [x1, y1, x2, y2]
                        scale_coords(img[si].shape[1:], tbox, [512, 640], ((1.25, 1.0), (0, 0)))
                        ious, _ = box_iou(predn[:, :4], tbox).max(1)
                        correct = (ious > 0.5).view(-1, 1)

                    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            else:
                # analog empty output
                bs, _, h, w = img.shape
                n_patches = 2
                num_anchors = 3
                nc = 1
                grid_h, grid_w = h // 32, w // 32
                placeholder_pred = torch.zeros((bs * n_patches, num_anchors, grid_h, grid_w, 5 + nc),
                                               device=img.device, dtype=torch.float32)
                loss, loss_items = compute_loss((placeholder_pred, None), targets, imgsz=img.shape)

                labels = targets[targets[:, 0] == 0, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                stats.append((torch.zeros(0, 1, dtype=torch.bool, device="cpu"),
                              torch.Tensor(), torch.Tensor(), tcls))

            total_loss += loss.item()
            losses += loss_items[:3]  # box, obj, cls


    evaluated_batches = min(len(dataloader), max_batches or len(dataloader))
    losses /= evaluated_batches
    total_loss /= evaluated_batches
    stats = [np.concatenate(x, 0) for x in zip(*stats)] if stats else []

    mp, mr, map50, map = 0., 0., 0., 0.
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names={0: 'UAV'})
        mp, mr, map50 = p.mean(), r.mean(), ap[:, 0].mean()

    print(f"Epoch {epoch}: Val Loss: {total_loss:.4f}, "
          f"Box: {losses[0]:.4f}, Obj: {losses[1]:.4f}, Cls: {losses[2]:.4f}, "
          f"Precision: {mp:.4f}, Recall: {mr:.4f}, mAP@0.5: {map50:.4f}")

    model.train()
    return total_loss, mp, mr, map50


def main(args):
    from dataset.dsloader_detect import Data
    from model.rbcn import RBCN
    from model.ecfnet import ECFNet
    from model.detect.loss import ComputeLoss

    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    print('testing device:', device)

    rbcn = RBCN(args, 1, if_denoise=True).to(device)
    if args.rbcn_weight_path:
        rbcn.load_state_dict(torch.load(args.rbcn_weight_path, map_location=device), strict=False)
        print(f"RBCN weights loaded from {args.rbcn_weight_path}")
    else:
        print("No RBCN weights supplied; using random initialization for the smoke test.")
    rbcn.requires_grad_(False)

    model = ECFNet(1, 3, rbcn, cfg=args.cfg, ch=64).to(device)
    if args.weight_path:
        model.load_state_dict(torch.load(args.weight_path, map_location=device), strict=False)
        print(f"ECFNet weights loaded from {args.weight_path}")
    else:
        print("No ECFNet weights supplied; using random initialization for the smoke test.")

    dataset = Data(args.dataset_path, args, args.test_split)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    compute_loss = ComputeLoss(model)
    test(
        model,
        dataloader,
        compute_loss,
        device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        width=args.imgsz,
        height=args.imgsz,
        save_labels=args.save_labels,
        max_batches=args.max_batches,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--norm', action='store_true', default=True)
    parser.add_argument('--num_patch', type=int, default=20)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--tokensz', type=int, default=4)
    parser.add_argument('--rbcn_block', choices=('c2fp', 'c3k2'), default='c2fp')
    parser.add_argument('--dev', type=str, default='cuda:0')
    parser.add_argument('--dataset_path', type=str, default='./data', help='dataset root containing images, masks, and labels')
    parser.add_argument('--test_split', type=str, default='test')
    parser.add_argument('--rbcn_weight_path', type=str, default=None)
    parser.add_argument('--weight_path', type=str, default=None, help='ECFNet checkpoint')
    parser.add_argument('--cfg', type=str, default='./cfg/head.yaml')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--conf_thres', type=float, default=0.5)
    parser.add_argument('--iou_thres', type=float, default=0.7)
    parser.add_argument('--save_labels', action='store_true', default=False)
    main(parser.parse_args())
