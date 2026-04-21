import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectAwareAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = max(1, channels // 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(nn.Conv2d(2, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, feat: torch.Tensor):
        ch = self.channel_att(feat) * feat
        mp = torch.max(ch, dim=1, keepdim=True)[0]
        ap = torch.mean(ch, dim=1, keepdim=True)
        sp = self.spatial_conv(torch.cat([mp, ap], dim=1))
        return ch * sp


class WeightedFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, channels, 1)
        self.conv_b = nn.Conv2d(channels, channels, 1)
        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        a_proj = self.conv_a(a)
        b_proj = self.conv_b(b)
        w = self.weight_conv(torch.cat([a_proj, b_proj], dim=1))
        return a_proj * w + b_proj * (1 - w)


class WDFFNet(nn.Module):
    def __init__(self, out_size: int = 256, pretrained: bool = True):
        super().__init__()
        self.back_a = timm.create_model("efficientnet_b0", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        self.back_b = timm.create_model("resnet50", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))

        ch_a = self.back_a.feature_info.channels()
        ch_b = self.back_b.feature_info.channels()
        target_ch = [64, 96, 128, 160]

        self.proj_a = nn.ModuleList([nn.Conv2d(ca, tc, 1) for ca, tc in zip(ch_a, target_ch)])
        self.proj_b = nn.ModuleList([nn.Conv2d(cb, tc, 1) for cb, tc in zip(ch_b, target_ch)])
        self.fusions = nn.ModuleList([WeightedFusion(tc) for tc in target_ch])
        self.oams = nn.ModuleList([ObjectAwareAttention(tc) for tc in target_ch])

        self.up3 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(96, 64, 2, stride=2)

        self.dec_conv3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.dec_conv2 = nn.Sequential(nn.Conv2d(192, 96, 3, padding=1), nn.ReLU(inplace=True))
        self.dec_conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(64, 1, 1)
        self.out_size = out_size
        self._printed_shapes = False

    def forward(self, x: torch.Tensor):
        feats_a = self.back_a(x)
        feats_b = self.back_b(x)
        fused = []
        for pa, pb, fa, fb, fusion, oam in zip(self.proj_a, self.proj_b, feats_a, feats_b, self.fusions, self.oams):
            f = fusion(pa(fa), pb(fb))
            f = oam(f)
            fused.append(f)

        f1, f2, f3, f4 = fused
        d3 = self.dec_conv3(torch.cat([self.up3(f4), f3], dim=1))
        d2 = self.dec_conv2(torch.cat([self.up2(d3), f2], dim=1))
        d1 = self.dec_conv1(torch.cat([self.up1(d2), f1], dim=1))
        logits = self.out_conv(d1)
        if logits.shape[2:] != (self.out_size, self.out_size):
            logits = F.interpolate(logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        if not self._printed_shapes:
            print(f"[WDFFNet] input={tuple(x.shape)} output={tuple(logits.shape)}")
            self._printed_shapes = True
        return logits
