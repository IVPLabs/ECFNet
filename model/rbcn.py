from model.blocks import BasicLayer, PixelUnshuffleDownsample, PatchEmbedWithResNet, Mlp, C3k2, Conv

import torch, math
import torch.nn as nn
from dataset.data_utils import generate_noisy_attention_mask
from model.attention import C2PSA



class RBCN(nn.Module):
    r""" Tile Selection Module
    Split image into non-overlapping tiles, 4*4/8*4..., classify each tile

    Args:
        args.
        num_cls: number of output classes (BCELoss-->1)
    """
    def __init__(self, args, num_cls, visual=False, if_denoise=False) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.if_denoise = if_denoise
        self.visual = visual
        self.num_patch = args.num_patch # the size of the final encoded representations, i.e. number of tiles (4*4)
        self.patch_embed = PatchEmbedWithResNet(img_size=args.imgsz,in_chans=3, embed_dim=args.dim, norm_layer=nn.BatchNorm2d)
        self.num_layers = int(math.log2((args.imgsz//args.tokensz)//args.num_patch))
        print(f"the num_layer is {self.num_layers}")

        patches_resolution = [args.imgsz//args.tokensz, args.imgsz//args.tokensz]
        depths = [2]*self.num_layers
        block_type = getattr(args, "rbcn_block", "c3k2")


        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):

            layer = BasicLayer(dim=int(args.dim * 2 ** i_layer),
                               hidden_dim=int(args.dim * 2 ** i_layer)//4,
                             input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                             depth=1,
                            downsample=True,
                            block_type=block_type,)
            self.layers.append(layer)


        self.norm = nn.ModuleList([nn.BatchNorm2d(args.dim * 2 ** (i + 1)) for i in range(self.num_layers)])
        self.linear = nn.ModuleList([nn.Linear(args.dim * (2 ** (i + 1)), num_cls) for i in range(self.num_layers)])

        self.downsample_from_160 = PixelUnshuffleDownsample(input_resolution=None, dim=128)
        self.downsample_from_80 = PixelUnshuffleDownsample(input_resolution=None, dim=256)
        self.downsample_from_40 = PixelUnshuffleDownsample(input_resolution=None, dim=512)

        self.conv_80_to_40 = Conv(c1=512, c2=512, s=2)
        self.conv_40_to_20 = Conv(c1=1024, c2=1024, s=2)
        self.conv_20_press = Conv(c1=2048, c2=1024, s=1)

        self.conv_press1 = Conv(c1=512, c2=256, s=2)
        self.conv_press2 = Conv(c1=1024, c2=256)
        self.conv_press3 = Conv(c1=1024, c2=256,)

        self.attention_block = C2PSA(c1=768, c2=768, n=2)
        self.conv_press4 = Conv(c1=768, c2=192)
        self.classfy = Conv(c1=192, c2=1)

        if self.if_denoise:
            self.denoise_conv_80_press = Conv(c1=513, c2=256, k=3)
            self.denoise_conv_80 = Conv(c1=256, c2=1, k=1)

            self.denoise_conv_40_press1 = Conv(c1=1025, c2=512, k=3)
            self.denoise_conv_40_press2 = Conv(c1=512, c2=256, k=3)
            self.denoise_conv_40 = Conv(c1=256, c2=1, k=1)

            self.denoise_conv_20_press1 = Conv(c1=2049, c2=1024, k=3)
            self.denoise_conv_20_press2 = Conv(c1=1024, c2=512, k=3)
            self.denoise_conv_20_press3 = Conv(c1=512, c2=256, k=3)
            self.denoise_conv_20 = Conv(c1=256, c2=1, k=1)

    def forward(self, x, gt_attention_mask=None, ecfnet=False):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        x_downsample = []
        x_downsample.append(x)


        for layer in self.layers:
            x = layer(x)
            if x.shape[3] < 320:
                x_downsample.append(x)


        feature_80 = torch.cat((self.downsample_from_160(x_downsample[0]), x_downsample[1]), dim=1)
        feature_40 = torch.cat((self.downsample_from_80(x_downsample[1]), x_downsample[2]), dim=1)
        feature_20 = torch.cat((self.downsample_from_40(x_downsample[2]), x_downsample[3]), dim=1)

        noise_mask = []

        if gt_attention_mask:
            if self.if_denoise:
                noise_attention_mask_80 = generate_noisy_attention_mask(gt_attention_mask[0], sigma=1)
                noise_attention_mask_40 = generate_noisy_attention_mask(gt_attention_mask[1], sigma=0.5)
                noise_attention_mask_20 = generate_noisy_attention_mask(gt_attention_mask[2], sigma=0.25)

                feature_noise_80 = torch.cat((feature_80, noise_attention_mask_80), dim=1)
                feature_noise_40 = torch.cat((feature_40, noise_attention_mask_40), dim=1)
                feature_noise_20 = torch.cat((feature_20, noise_attention_mask_20), dim=1)

                noise_mask_80 = self.denoise_conv_80(self.denoise_conv_80_press(feature_noise_80))
                noise_mask_40 = self.denoise_conv_40(self.denoise_conv_40_press2(self.denoise_conv_40_press1(feature_noise_40)))
                noise_mask_20 = self.denoise_conv_20(self.denoise_conv_20_press3(self.denoise_conv_20_press2(self.denoise_conv_20_press1(feature_noise_20))))
                noise_mask = [noise_mask_80, noise_mask_40, noise_mask_20]

            feature_80 = feature_80 * gt_attention_mask[0]
            feature_40 = feature_40 * gt_attention_mask[1]
            feature_20 = feature_20 * gt_attention_mask[2]


        feature_80 = self.conv_80_to_40(feature_80)
        feature_40 = self.conv_40_to_20(feature_40)
        feature_20 = self.conv_20_press(feature_20)

        if self.visual:
            x_downsample.append(feature_80)
            x_downsample.append(feature_40)
            x_downsample.append(feature_20)
            x_downsample.append(self.conv_press1(feature_80))
            x_downsample.append(self.conv_press2(feature_40))
            x_downsample.append(self.conv_press3(feature_20))

        feature = torch.cat([self.conv_press1(feature_80), self.conv_press2(feature_40), self.conv_press3(feature_20)], dim=1)
        mask = self.classfy(self.conv_press4(self.attention_block(feature)))


        if ecfnet:
            return x_downsample[0], x_downsample[1], x_downsample[2], x_downsample[3], mask
        else:
            if self.if_denoise:
                return mask, noise_mask
            return mask
