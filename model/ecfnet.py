import torch
import time
from .detect.head import Detect
from .detect.common import *
from .blocks import *
import logging
from copy import deepcopy
from pathlib import Path




class ECFNet(nn.Module):
    def __init__(self, nc, anchors, rbcn, cfg, ch, use_apkd=True,
                 train_patch_num=6):
        super(ECFNet, self).__init__()
        self.rbcn = rbcn
        self.use_apkd = use_apkd

        # Same selector as HeatMapParser in the reference project; only the
        # local class name differs.
        self.expsilcer_4 = ExpSlicer(
            128, 8, 0.5, train_target_num=train_patch_num
        )


        self.head0_upsample = nn.Upsample(None, 2, 'nearest')
        # Concat
        self.head0_c3k2 = C3k2(c1=1536, c2=512, n=2, shortcut=True, g=1, e=0.25)

        self.head1_upsample = nn.Upsample(None, 2, 'nearest')
        #Concat
        self.head1_c3k2 = C3k2(c1=768, c2=256, n=2, shortcut=True, g=1, e=0.25)

        self.head2_upsample = nn.Upsample(None, 2, 'nearest')
        # Concat
        self.head2_c3k2 = C3k2(c1=384, c2=128, n=2, shortcut=True, g=1, e=0.25)

        self.head3_conv = Conv(128, 128, 3, 2)
        # Concat
        self.head3_c3k2 = C3k2(c1=384, c2=256, n=2, shortcut=True, g=1, e=0.25)

        # APKD is the local counterpart of the reference ContMixGuidance.
        # Each level turns (teacher, student) features into a guided student
        # representation before the feature-distillation loss is evaluated.
        if self.use_apkd:
            self.apkd_0 = APKD(dim=512)
            self.apkd_1 = APKD(dim=256)
            self.apkd_2 = APKD(dim=128)
            self.apkd_3 = APKD(dim=256)

        # ===========================Head============================================
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            self.yaml['anchors'] = round(anchors)  # override yaml value

        self.detect_head = parse_model(deepcopy(self.yaml), ch=[ch])

        if isinstance(self.detect_head, Detect):
            s = 256  # 2x min stride
            self.detect_head.inplace = True
            # TODO
            self.detect_head.stride = torch.tensor([4., 8.])
            self.detect_head.anchors /= self.detect_head.stride.view(-1, 1, 1)
            self.stride = self.detect_head.stride


    def forward(self, x, teacher_features=None):
        kd_features = []
        feat_4, feat_8, feat_16, feat_32, pred_masks = self.rbcn(x, ecfnet=True)


        pred_masks = (
            torch.sigmoid(pred_masks) >= self.expsilcer_4.threshold
        ).to(dtype=pred_masks.dtype)

        feat_4_full = feat_4
        feat_4, offsets = self.expsilcer_4((feat_4, pred_masks))


        if len(feat_4) == 0:
            return (None, (None, None)), pred_masks


        _, feat_8, feat_16, feat_32 = extract_features(
            [feat_4_full, feat_8, feat_16, feat_32],
            offsets,
            ratios=[1, 2, 4, 8],
        )


        if teacher_features is not None:
            teacher_features_0, teacher_features_1, teacher_features_2, teacher_features_3 = extract_features(teacher_features, offsets, ratios=[4, 2, 1, 2])
            teacher_features = [teacher_features_0,teacher_features_1, teacher_features_2, teacher_features_3]


        feats = []
        x = self.head0_upsample(feat_32)  # 1024
        x = torch.cat([x, feat_16], dim=1)  # 1024+512=1536
        x = self.head0_c3k2(x)  # 512
        kd_features.append(x)


        x = self.head1_upsample(x)  #512
        x = torch.cat([x, feat_8], dim=1) #1024
        x = self.head1_c3k2(x) #256
        kd_features.append(x)


        temp_256 = x

        x = self.head2_upsample(x)  # 256
        x = torch.cat([x, feat_4], dim=1)  # 384
        x = self.head2_c3k2(x)  # 128
        feats.append(x)
        kd_features.append(x)


        x = self.head3_conv(x)  #128
        x = torch.cat([x, temp_256], dim=1)  #384
        x = self.head3_c3k2(x)  #256
        feats.append(x)
        kd_features.append(x)



        x = (feats, offsets, pred_masks)
        x = self.detect_head(x)

        if teacher_features is not None and self.use_apkd:
            kd_features[0] = self.apkd_0(teacher_features[0], kd_features[0])
            kd_features[1] = self.apkd_1(teacher_features[1], kd_features[1])
            kd_features[2] = self.apkd_2(teacher_features[2], kd_features[2])
            kd_features[3] = self.apkd_3(teacher_features[3], kd_features[3])

        if teacher_features is not None:
            return x, pred_masks, kd_features, teacher_features
        else:
            return x, pred_masks

def parse_model(d, ch):  # model_dict, input_channels(3)
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']

    anchors = [[5, 6, 8, 15, 17, 11], [10,13, 16,30, 33,23]]
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m is Detect:
            if len(args) > 1 and isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            args.append([128, 256])
        m_ = m(*args)  # module
    return m_


