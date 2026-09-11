
import os
import argparse
from sklearn.metrics import confusion_matrix, precision_recall_curve, mean_absolute_error
import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
import numpy as np
from torch import optim
import utils.utils as utils
from dataset.dsloader import Data
from model.rbcn import RBCN
from PIL import Image
import cv2
import time
from openpyxl import Workbook
from model.loss import FocalLoss, bce_dice_iou_loss

def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer

def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.training_excel_path)), exist_ok=True)
    workbook, sheet = utils.create_Excel(args.training_excel_path)
    print(args)

    # load data
    print('============================ loading data ============================')
    root = args.dataset_path

    dataset_tr = Data(root, args, args.train_split, gt_attention_aug=True)
    dataset_te = Data(root, args, args.val_split)

    train_loader = DataLoader(dataset_tr, args.batchsz, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(dataset_te, args.batchsz, num_workers=args.num_workers, shuffle=True)

    # check cuda
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    print('training device:', device)

    # build model
    num_ch = 3
    num_cls = 1 if args.loss == 'bce' else 2
    model = RBCN(args, num_cls, if_denoise=True)
    num_layers = model.num_layers
    model = model.to(device)

    if args.weight_path:
        if os.path.exists(args.weight_path):
            checkpoint = torch.load(args.weight_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print(f"Pretrained weights loaded from {args.weight_path}")
        else:
            print(f"Pretrained weights not found at {args.weight_path}, skipping...")


    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )
    weight_save_dir = args.output_dir
    os.makedirs(weight_save_dir, exist_ok=True)


    # train and validate
    print('============================ Model Training ============================')
    train_loss, test_loss = 0.0, 0.0
    tn, fp, fn, tp = 0.0, 0.0, 0.0, 0.0
    cm = np.zeros((2, 2))
    max_P = 0
    max_R = 0
    for epoch in range(args.epoch):
        model.train()
        # train model
        if (epoch+1)%15 == 0:
            optimizer = adjust_learning_rate(optimizer, decay_rate=0.1)
            
        for batch_i, (xtr, ytr, gt_attention_mask) in enumerate(train_loader):
            if args.max_train_batches and batch_i >= args.max_train_batches:
                break
            # generate GT labels 20*20
            atr = nn.MaxPool2d(kernel_size=(args.imgsz // args.num_patch), stride=(args.imgsz // args.num_patch))(ytr)
            atr_40 = nn.MaxPool2d(kernel_size=(args.imgsz // (args.num_patch * 2)),
                                stride=(args.imgsz // (args.num_patch * 2)))(ytr)
            atr_80 = nn.MaxPool2d(kernel_size=(args.imgsz // (args.num_patch * 4)),
                                stride=(args.imgsz // (args.num_patch * 4)))(ytr)

            xtr, ytr, atr, atr_40, atr_80 = xtr.to(device), ytr.to(device), atr.to(device), atr_40.to(device), atr_80.to(device)

            gt_attention_mask = [attention_mask.to(device) for attention_mask in gt_attention_mask]

            optimizer.zero_grad()

            ptr, noise_ptr = model(xtr, gt_attention_mask)

            noise_ltr1 = bce_dice_iou_loss(
                noise_ptr[0], atr_80, pos_weight=args.pos_weight,
                dice_weight=args.dice_weight, iou_weight=args.iou_weight,
            )
            noise_ltr2 = bce_dice_iou_loss(
                noise_ptr[1], atr_40, pos_weight=args.pos_weight,
                dice_weight=args.dice_weight, iou_weight=args.iou_weight,
            )
            noise_ltr3 = bce_dice_iou_loss(
                noise_ptr[2], atr, pos_weight=args.pos_weight,
                dice_weight=args.dice_weight, iou_weight=args.iou_weight,
            )
            noise_ltr_all = noise_ltr1 + noise_ltr2 + noise_ltr3

            coarse_ltr = bce_dice_iou_loss(
                ptr, atr, pos_weight=args.pos_weight,
                dice_weight=args.dice_weight, iou_weight=args.iou_weight,
            )
            ltr = coarse_ltr + args.noise_loss_weight * noise_ltr_all

            ltr.backward()
            optimizer.step()

            train_loss += ltr.item()

        # evaluate model
        pred = np.zeros((0,1,args.num_patch,args.num_patch))
        gt = np.zeros((0,1,args.num_patch,args.num_patch))
        # The reference run.py validates while the model remains in training
        # mode. Keep that behavior by default for reproducibility, while
        # allowing conventional eval-mode validation explicitly.
        if not getattr(args, 'reference_validation_train_mode', True):
            model.eval()
        with torch.no_grad():
            for batch_i, (xte, yte, name) in enumerate(test_loader):
                if args.max_val_batches and batch_i >= args.max_val_batches:
                    break
                ate = nn.MaxPool2d(kernel_size=(args.imgsz//args.num_patch), stride=(args.imgsz//args.num_patch))(yte)
                xte, yte, ate = xte.to(device), yte.to(device), ate.to(device)

                pte, _ = model(xte)

                lte = bce_dice_iou_loss(
                    pte, ate, pos_weight=args.pos_weight,
                    dice_weight=args.dice_weight, iou_weight=args.iou_weight,
                )
                test_loss += lte.item()
                pte = nn.Sigmoid()(pte)
                if args.loss != 'bce':
                    pte = torch.unsqueeze(torch.argmax(pte, 1), 1)
                pte = pte.cpu().numpy()
                ate = ate.cpu().numpy()

                ate[ate>=0.5] = 1
                ate[ate<0.5] = 0

                pred = np.append(pred, pte, axis=0)
                gt = np.append(gt, ate, axis=0)
                pte[pte >= args.th] = 1 # adjustable threshold
                pte[pte < args.th] = 0
                cm += confusion_matrix(ate.astype(np.int32).flatten(), pte.flatten())

        pred = pred.flatten()
        gt = gt.flatten()
        precision, recall, threshold = precision_recall_curve(gt, pred)
        f_scores = 1.3*recall*precision/(recall+0.3*precision + 1e-20)

        mae = mean_absolute_error(gt, pred)
        tn, fp, fn, tp = cm.ravel()

        # Print metrics
        train_batches = min(len(train_loader), args.max_train_batches or len(train_loader))
        val_batches = min(len(test_loader), args.max_val_batches or len(test_loader))
        train_loss_avg = train_loss / train_batches
        test_loss_avg = test_loss / val_batches
        max_f_score = np.max(f_scores)
        iou = tp / (tp + fn + fp + 1e-20)
        tpr = tp / (tp + fn + 1e-20)
        if iou > max_P:
            weight_path = os.path.join(weight_save_dir, f"model_epoch_{epoch + 1}_bestP.pt")
            torch.save(model.state_dict(), weight_path)
            max_P = iou
            print(f"Model weights saved to {weight_path}")
        if tpr > max_R:
            weight_path = os.path.join(weight_save_dir, f"model_epoch_{epoch + 1}_bestR.pt")
            torch.save(model.state_dict(), weight_path)
            max_R = tpr
            print(f"Model weights saved to {weight_path}")
        print(
            f"Epoch {epoch + 1}\tTrain Loss: {train_loss_avg:.4f}\tTest Loss: {test_loss_avg:.4f}"
            f"\tTP: {tp:.4f}\tFN: {fn:.4f}\tFP: {fp:.4f}\tTN: {tn:.4f}"
            f"\tMAE: {mae:.4f}\tMax F-score: {max_f_score:.4f}\tIoU: {iou:.4f}\tTPR: {tpr:.4f}"
        )

        # Write metrics to Excel
        row = [epoch + 1, train_loss_avg, test_loss_avg, tp, fn, fp, tn, mae, max_f_score, iou, tpr]
        sheet.append(row)

        # Save the Excel file after each epoch
        workbook.save(args.training_excel_path)

        cm = np.zeros((2, 2))
        train_loss = 0.0
        test_loss = 0.0
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, help='image size', default=640) # original image size
    argparser.add_argument('--aug', action=argparse.BooleanOptionalAction, help='data augmentation or not', default=True)
    argparser.add_argument('--norm', action=argparse.BooleanOptionalAction, help='normalize data or not', default=True)
    argparser.add_argument('--num_patch', type=int, help='number of patches (eg. 4*4 or 8*8)', default=20)

    argparser.add_argument('--dim', type=int, help='attention embedding dimension for patch selection', default=128)
    argparser.add_argument('--tokensz', type=int, help='token size for image embedding', default=4)
    argparser.add_argument(
        '--rbcn_block',
        type=str,
        choices=('c2fp', 'c3k2'),
        default='c3k2',
        help='RBCN basic block: reference C3k2 (default) or paper C2fP',
    )
    argparser.add_argument('--th', type=float, help='threshold for attention or not', default=0.5)
    argparser.add_argument('--dev', type=str, help='cuda device', default='cuda:2')
    argparser.add_argument('--epoch', type=int, help='number of training epochs', default=100)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--weight_decay', type=float, help='Adam weight decay', default=0.0)
    argparser.add_argument('--batchsz', type=int, help='batch size', default=12)
    argparser.add_argument('--loss', type=str, help='loss function(bce/ce)', default='bce')
    argparser.add_argument('--beta', type=float, help='weighted cross entropy parameter', default=2)
    argparser.add_argument('--pos_weight', type=float, default=1.0)
    argparser.add_argument('--dice_weight', type=float, default=1.0)
    argparser.add_argument('--iou_weight', type=float, default=1.0)
    argparser.add_argument('--noise_loss_weight', type=float, default=0.3)
    argparser.add_argument('--weight_path', type=str, default=None, help='optional pretrained RBCN checkpoint')

    # patch selection
    argparser.add_argument('--training_excel_path', type=str, default='./runs/rbcn/training_metrics.xlsx', help='save training loss Excel path')
    argparser.add_argument('--dataset_path', type=str, default='./data', help='dataset root containing images, masks, and labels')
    argparser.add_argument('--train_split', type=str, default='train')
    argparser.add_argument('--val_split', type=str, default='test')
    argparser.add_argument('--num_workers', type=int, default=4)
    argparser.add_argument(
        '--reference-validation-train-mode',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='match run.py by validating in train mode; use --no-reference-validation-train-mode for standard eval mode',
    )
    argparser.add_argument('--max_train_batches', type=int, default=0, help='0 uses the complete training set')
    argparser.add_argument('--max_val_batches', type=int, default=0, help='0 uses the complete validation set')
    argparser.add_argument('--output_dir', type=str, default='./runs/rbcn/weights', help='directory used to save RBCN checkpoints')
    args = argparser.parse_args()
    main(args)
