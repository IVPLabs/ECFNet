import argparse
import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
import numpy as np
from torch import optim
import utils.utils as utils
from dataset.dsloader_detect import Data, custom_collate_fn
from model.rbcn import RBCN
from PIL import Image
import time
from openpyxl import Workbook
from torch.cuda import amp
from model.ecfnet import ECFNet
from model.detect.loss import ComputeLoss
import test_ECFNet as test
from torch.optim import lr_scheduler
import math
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def one_cycle(y1=0.01, y2=1.0, epochs=100):
    return lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (y1 - y2) + y2


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
        break
    return optimizer


def load_ecfnet_state(model, checkpoint, strict=True):
    """Load checkpoints across the patch_selector -> rbcn naming change."""
    if any(key.startswith('patch_selector.') for key in checkpoint):
        checkpoint = {
            ('rbcn.' + key[len('patch_selector.'):]
             if key.startswith('patch_selector.') else key): value
            for key, value in checkpoint.items()
        }
    model.load_state_dict(checkpoint, strict=strict)

def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.training_excel_path)), exist_ok=True)
    workbook, sheet = utils.create_Excel(args.training_excel_path)
    print(args)

    use_teacher = bool(args.teacher_weight_path and os.path.isfile(args.teacher_weight_path))
    if args.teacher_weight_path and not use_teacher:
        print(
            f"Teacher weights not found at {args.teacher_weight_path}; "
            "continuing without APKD knowledge distillation."
        )
    elif not args.teacher_weight_path:
        print("No teacher weights supplied; training without APKD knowledge distillation.")
    args.use_teacher = use_teacher

    # load data
    print('============================ loading data ============================')
    root = args.dataset_path

    dataset_tr = Data(root, args, args.train_split)
    dataset_te = Data(root, args, args.val_split)
    train_loader = DataLoader(dataset_tr, args.batchsz, num_workers=args.num_workers, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(dataset_te, 1, num_workers=args.num_workers, shuffle=True)

    # check cuda
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    print('training device:', device)

    # build model
    num_ch = 3
    num_cls = 1 if args.loss == 'bce' else 2
    rbcn = RBCN(args, num_cls, if_denoise=True)
    num_layers = rbcn.num_layers
    rbcn = rbcn.to(device)
    if args.rbcn_weight_path:
        checkpoint = torch.load(args.rbcn_weight_path, map_location=device)
        rbcn.load_state_dict(checkpoint)
        print(f"RBCN weights loaded from {args.rbcn_weight_path}")
    else:
        print("No RBCN weights supplied; using random initialization for the smoke test.")
    rbcn.to(device)
    rbcn.requires_grad_(False)  #Freeze the parameters of RBCN

    if args.pretrained_weight_path:
        if os.path.exists(args.pretrained_weight_path):
            checkpoint = torch.load(args.pretrained_weight_path, map_location=device)
            rbcn.load_state_dict(checkpoint, strict=False)
            print(f"Pretrained weights loaded from {args.pretrained_weight_path}")
        else:
            print(f"Pretrained weights not found at {args.pretrained_weight_path}, skipping...")


    model = ECFNet(
        1, 3, rbcn, cfg=args.cfg_path, ch=64,
        use_apkd=use_teacher, train_patch_num=args.train_patch_num,
    )
    model = model.to(device)
    compute_loss = ComputeLoss(model)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.937, nesterov=True
    )
    nb = len(train_loader)
    nw = max(round(3.0 * nb), 100)
    nbs = 64

    weight_save_dir = args.output_dir
    os.makedirs(weight_save_dir, exist_ok=True)

    if args.weight_path:
        checkpoint = torch.load(args.weight_path, map_location=device)
        load_ecfnet_state(model, checkpoint, strict=use_teacher)
        model.to(device)
        model.rbcn = rbcn
        model.rbcn.requires_grad_(False) #Freeze the parameters of RBCN

    teacher_model = None
    yolo_features = {}
    target_layers = (13, 16, 19, 22)
    if use_teacher:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required when teacher weights are supplied"
            ) from exc

        teacher_model = YOLO(model=args.teacher_weight_path).to(device)
        teacher_model.model.eval()

        def save_yolo_feature(name):
            def hook(module, inputs, output):
                yolo_features[name] = output
            return hook

        for index in target_layers:
            teacher_model.model.model[index].register_forward_hook(
                save_yolo_feature(f'layer_{index}')
            )

    # train and validate
    print('============================ Model Training ============================')
    for epoch in range(args.epoch):

        one_epoch_loss = torch.zeros(4, device=device)  # box, obj, cls, kd

        if (epoch + 1) % 40 == 0:
            optimizer = adjust_learning_rate(optimizer, decay_rate=0.1)

        update_lr_group = optimizer.param_groups
        print('before lr: ', update_lr_group[0]['lr'])

        for i, (xtr, ytr, labels, shapes, teacher_paths) in enumerate(train_loader):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            model.train()
            model.rbcn.eval()
            xtr, ytr, labels = xtr.to(device), ytr.to(device), labels.to(device)
            if use_teacher:
                yolo_features.clear()
                with torch.no_grad():
                    teacher_model.predict(
                        source=list(teacher_paths),
                        imgsz=(args.imgsz, args.imgsz),
                        device=device,
                        verbose=False,
                        save=False,
                    )
                missing_layers = [
                    index for index in target_layers
                    if f'layer_{index}' not in yolo_features
                ]
                if missing_layers:
                    raise RuntimeError(
                        f"Teacher hooks captured no features for layers {missing_layers}"
                    )
                features = [
                    yolo_features[f'layer_{index}'].detach().clone()
                    for index in target_layers
                ]
                det_output, pred, stu_features, teacher_features = model(xtr, features)
                kd_items = (stu_features, teacher_features)
            else:
                det_output, pred = model(xtr)
                kd_items = None

            x, patch_offsets = det_output

            if patch_offsets is not None:
                loss, loss_items = compute_loss((x, patch_offsets), labels, kd_items, imgsz=xtr.shape)
            else:
                loss, loss_items = compute_loss(x, labels, kd_items, imgsz=xtr.shape)

            one_epoch_loss += loss_items[:4]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_batches = min(len(train_loader), args.max_train_batches or len(train_loader))
        one_epoch_loss /= train_batches

        print(f"Epoch {epoch}: Train  Box {one_epoch_loss[0]} Obj {one_epoch_loss[1]} Cls {one_epoch_loss[2]} KD {one_epoch_loss[3]}")

        val_loss, precision, recall, map50 = test.test(
            model, test_loader, compute_loss, device, epoch=epoch, num_classes=num_cls,conf_thres=0.001,
            save_labels=False, max_batches=args.max_val_batches)

        if (epoch+1) % 20 == 0:
            weight_path = os.path.join(weight_save_dir, f"epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), weight_path)
            print(f"Model weights saved to {weight_path}")




if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, help='image size', default=640)
    argparser.add_argument('--aug', action='store_true', help='data augmentation or not', default=False)
    argparser.add_argument('--norm', action='store_true', help='normalize data or not', default=True)
    argparser.add_argument('--num_patch', type=int, help='number of patches (eg. 4*4 or 8*8)', default=20)
    argparser.add_argument(
        '--train_patch_num', type=int, default=6,
        help='fixed number of selected patches per image during training',
    )

    argparser.add_argument('--dim', type=int, help='attention embedding dimension for patch selection', default=128)
    argparser.add_argument('--tokensz', type=int, help='token size for image embedding', default=4)
    argparser.add_argument(
        '--rbcn_block',
        type=str,
        choices=('c2fp', 'c3k2'),
        default='c3k2',
        help='RBCN basic block matching the reference PatchSelector',
    )
    argparser.add_argument('--th', type=float, help='threshold for attention or not', default=0.5)
    argparser.add_argument('--dev', type=str, help='cuda device', default='cuda:3')
    argparser.add_argument('--epoch', type=int, help='number of training epochs', default=300)
    argparser.add_argument('--lr', type=float, help='SGD learning rate', default=0.001)
    argparser.add_argument('--weight_decay', type=float, default=0.0)
    argparser.add_argument('--batchsz', type=int, help='batch size', default=16)
    argparser.add_argument('--loss', type=str, help='loss function(bce/ce)', default='bce')
    argparser.add_argument('--beta', type=float, help='weighted cross entropy parameter', default=2)
    argparser.add_argument('--pretrained_weight_path', type=str, default=None, help='optional additional RBCN checkpoint')
    argparser.add_argument('--weight_path', type=str, default=None, help='optional ECFNet checkpoint used to resume training')
    argparser.add_argument(
        '--rbcn_weight_path', type=str, default=None,
        help='trained coarse-stage RBCN checkpoint',
    )
    argparser.add_argument(
        '--teacher_weight_path', type=str, default=None,
        help='optional teacher checkpoint for APKD',
    )
    argparser.add_argument(
        '--teacher_dataset_path', type=str, default=None,
        help='optional directory containing teacher training images',
    )
    argparser.add_argument('--training_excel_path', type=str, default='./runs/ecfnet/training_metrics.xlsx')
    argparser.add_argument('--dataset_path', type=str, default='./data', help='dataset root containing images, masks, and labels')
    argparser.add_argument('--cfg_path', type=str, default='./cfg/head.yaml')
    argparser.add_argument('--train_split', type=str, default='train')
    argparser.add_argument('--val_split', type=str, default='test')
    argparser.add_argument('--num_workers', type=int, default=4)
    argparser.add_argument('--max_train_batches', type=int, default=0)
    argparser.add_argument('--max_val_batches', type=int, default=0)
    argparser.add_argument('--output_dir', type=str, default='./runs/ecfnet/weights')


    args = argparser.parse_args()
    main(args)
