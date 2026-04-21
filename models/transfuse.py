import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFusionBlock(nn.Module):
    def __init__(self, cnn_ch: int, trans_ch: int, out_ch: int):
        super().__init__()
        self.conv_cnn = nn.Conv2d(cnn_ch, out_ch, 1)
        self.conv_trans = nn.Conv2d(trans_ch, out_ch, 1)
        self.conv_out = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        if trans_feat.ndim == 4 and trans_feat.shape[-1] > trans_feat.shape[1]:
            trans_feat = trans_feat.permute(0, 3, 1, 2)
        x = self.conv_cnn(cnn_feat) + self.conv_trans(trans_feat)
        return self.act(self.conv_out(x))


class TransFuse(nn.Module):
    def __init__(self, out_size: int = 256, pretrained: bool = True):
        super().__init__()
        self.cnn = timm.create_model("efficientnet_b0", pretrained=pretrained, features_only=True)
        self.trans = timm.create_model("mobilenetv3_large_100", pretrained=pretrained, features_only=True)

        c_ch = self.cnn.feature_info.channels()
        t_ch = self.trans.feature_info.channels()

        self.fuse3 = BiFusionBlock(c_ch[-1], t_ch[-1], 256)
        self.fuse2 = BiFusionBlock(c_ch[-2], t_ch[-2], 128)
        self.fuse1 = BiFusionBlock(c_ch[-3], t_ch[-3], 64)
        self.conv_final = nn.Conv2d(256 + 128 + 64, 1, 1)
        self.out_size = out_size
        self._printed_shapes = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feats = self.cnn(x)
        trans_feats = self.trans(x)

        f3 = self.fuse3(cnn_feats[-1], trans_feats[-1])
        f2 = self.fuse2(cnn_feats[-2], trans_feats[-2])
        f1 = self.fuse1(cnn_feats[-3], trans_feats[-3])

        f3_up = F.interpolate(f3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        logits = self.conv_final(torch.cat([f3_up, f2_up, f1], dim=1))

        if logits.shape[2:] != (self.out_size, self.out_size):
            logits = F.interpolate(logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        if not self._printed_shapes:
            print(f"[TransFuse] input={tuple(x.shape)} output={tuple(logits.shape)}")
            self._printed_shapes = True
        return logits
