# Loss functions
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.general import bbox_iou


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device

        h = {
            'box': 0.05,
            'obj': 1.0,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj_pw': 1.0,
            'fl_gamma': 0,
            'label_smoothing': 0.0,  #
            'anchor_t': 4.0,
        }

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))


        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        # run_for_detect.py leaves focal/QFocal wrapping disabled.

        det = model.detect_head  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'anchor_grid', 'stride':
            setattr(self, k, getattr(det, k))
        self.neg_anchor_iou_thres = 0.7
        self.pos_anchor_iou_thres = 0.15
        self.pos_anchor_num = 4
        self.lpixl_critreia = None
        self.anchors = det.anchors



    def __call__(self, p, targets, kd_items=None, imgsz=None, masks=None, m_weights=None):  # predictions, targets, model
        p_det = p
        offsets = []
        device = targets.device

        if kd_items is not None:
            student_fpn, teacher_fpn = kd_items
            lkd = self.plain_kd_loss(student_fpn, teacher_fpn, alpha=1.0)
        else:
            lkd = torch.zeros(1, device=device)

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)


        if targets.dim() == 3 and targets.shape[1] == 1:  # [bs, 1, 6]
            targets = targets.squeeze(1)  # [bs, 6]
        elif targets.dim() == 3:  # [bs, nt, 6]
            targets = targets.view(-1, 6)  # [bs * nt, 6]

        if p_det is not None and p_det[0] is not None and p_det[1] is not None:  # stupid
            # ta = time_synchronized()
            if isinstance(p_det, tuple):
                p, offsets = p_det
                tcls, tbox, indices, anchors = self.build_patch_targets(offsets, targets, imgsz)  # targets

            # Losses
            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]

                    # Regression
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box

                    iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss


                    # Objectness
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                    # Classification
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                obji = F.binary_cross_entropy_with_logits(
                    pi[..., 4].clamp_(-9.21, 9.21), tobj, pos_weight=torch.tensor([2], device=device), reduction="mean"
                )

                lobj += obji * self.balance[i]  # obj loss

                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()


        bs = targets[0].shape[0] if targets is not None else tobj.shape[0]
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkd *= 1.0

        loss = (lbox + lobj + lcls + lkd) * 1.0
        loss_items = torch.cat((lbox, lobj, lcls, lkd, loss)).detach()

        return loss * bs, loss_items

    def build_patch_targets(self, patch_offsets, targets, imgsz):  # for fast-mode, fixed patch division
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]
        dtype, device = targets.dtype, targets.device

        tcls, tbox, indices, anch = [], [], [], []
        bs, _, height, width = imgsz

        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)
        bi_ = torch.arange(patch_offsets[0].shape[0], device=device)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            ], device=device).float() * g  # offsets

        for i in range(self.nl):
            patch_off = patch_offsets[i]
            anchors = self.anchors[i]


            r = (2 ** (i - 1)) if self.nl == 4 else 2 ** i
            gain[2:6] = torch.tensor([width, height, width, height], dtype=dtype) / (4 * r)  # TODO: from 4 to 32
            grid_wh = patch_off[:1, [3, 4]] - patch_off[:1, [1, 2]]
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter, shape(nt_, 7)

                tb, txc, tyc = t[:, [0, 2, 3]].chunk(3, dim=1)
                pb, px1, py1, px2, py2 = (patch_off.T).chunk(5, dim=0)  # shape(1,m)

                contained = (tb == pb) & (txc > px1 - g) & (txc < px2 - g) & (tyc > py1 - g) & (tyc < py2 - g)  # shape(n,m)
                ti, pj = torch.nonzero(contained)   .T  # i-th target is contained within j-th patch
                t = t[ti]  # shape(n,7)
                
                # Offsets
                gxy = t[:, 2:4]
                gxi = grid_wh - gxy  # inverse
                j, k = ((gxy - gxy.floor() < g) & (gxy > 0.-g)).T
                l, m = ((gxi - gxi.floor() < g) & (gxi > 1.-g)).T
                # j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                
                t[:, 0] = bi_[pj]  # converted batch-indices
                t[:, 2:4] -= patch_off[pj, 1:3]  # converted xc, yc (minus px1, py1)

                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        b, a, gi, gj = indices[0]

        return tcls, tbox, indices, anch


    def kd_feature_loss(self, kd_features, teacher_features, mode='l2', normalize=False, weights=None):
        """Reference feature-distillation loss: sum over the four FPN levels."""
        assert len(kd_features) == len(teacher_features), "feature level count mismatch"
        total_loss = torch.tensor(0.0, device=kd_features[0].device)
        for i, (student, teacher) in enumerate(zip(kd_features, teacher_features)):
            if normalize:
                student = F.normalize(student, dim=1)
                teacher = F.normalize(teacher, dim=1)
            if mode == 'l2':
                loss = F.mse_loss(student, teacher)
            elif mode == 'cosine':
                student = student.view(student.size(0), student.size(1), -1)
                teacher = teacher.view(teacher.size(0), teacher.size(1), -1)
                loss = 1 - F.cosine_similarity(student, teacher, dim=1).mean()
            else:
                raise ValueError(f"Unsupported distillation mode: {mode}")
            total_loss += (weights[i] if weights else 1.0) * loss
        return total_loss.unsqueeze(0)

    def plain_kd_loss(self, student_fpn, teacher_fpn, alpha=1.0):
        """Mean four-level MSE used after APKD guidance in the reference."""
        if len(student_fpn) != len(teacher_fpn):
            raise ValueError(
                f"Student/teacher feature count mismatch: "
                f"{len(student_fpn)} != {len(teacher_fpn)}"
            )
        if not student_fpn:
            raise ValueError("KD feature lists must not be empty")

        total_loss = student_fpn[0].new_zeros(())
        for student_feat, teacher_feat in zip(student_fpn, teacher_fpn):
            if student_feat.shape != teacher_feat.shape:
                raise ValueError(
                    "APKD output and teacher feature shapes differ: "
                    f"{tuple(student_feat.shape)} != {tuple(teacher_feat.shape)}"
                )
            total_loss += F.mse_loss(student_feat, teacher_feat)
        return (alpha * total_loss / len(student_fpn)).unsqueeze(0)

    @staticmethod
    def dice_loss(inputs, targets):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

    @staticmethod
    def quality_dice_loss(inputs, targets, weight=None, gamma: float = 2):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        if weight is not None:
            weight = weight.flatten(1)
            inputs = inputs * weight
            targets = targets * weight

        numerator = 2 * (inputs - targets).abs().sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = (numerator + 1) / (denominator + 1)
        return loss.mean()

    @staticmethod
    def sigmoid_quality_focal_loss(inputs, targets, weight=None, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction="none")
        loss = ce_loss * ((prob - targets).abs() ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()
