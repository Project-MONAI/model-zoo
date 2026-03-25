import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from model import ConvBNReLU, scSE, AttentionGate


class TransformerEncoder(nn.Module):
	def __init__(self, backbone: str = "swin_tiny_patch4_window7_224", in_ch: int = 1, pretrained: bool = True):
		super().__init__()
		self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, in_chans=in_ch)
		feat_info = self.backbone.feature_info
		self.channels = [s['num_chs'] for s in feat_info]

	def forward(self, x):
		feats = self.backbone(x)
		return feats  # list [C1@H/4, C2@H/8, C3@H/16, C4@H/32]


class AttentionUNetTransformer(nn.Module):
	def __init__(self, in_ch: int = 1, out_ch: int = 3, backbone: str = "swin_tiny_patch4_window7_224", pretrained: bool = True):
		super().__init__()
		self.enc = TransformerEncoder(backbone=backbone, in_ch=in_ch, pretrained=pretrained)
		c1, c2, c3, c4 = self.enc.channels

		# project features to stable decoder widths
		dec_c4 = 384
		dec_c3 = 192
		dec_c2 = 96
		dec_c1 = 48

		self.proj4 = nn.Conv2d(c4, dec_c4, 1, bias=False)
		self.proj3 = nn.Conv2d(c3, dec_c3, 1, bias=False)
		self.proj2 = nn.Conv2d(c2, dec_c2, 1, bias=False)
		self.proj1 = nn.Conv2d(c1, dec_c1, 1, bias=False)

		self.dec3 = self._decoder_block(dec_c4, dec_c3)
		self.dec2 = self._decoder_block(dec_c3, dec_c2)
		self.dec1 = self._decoder_block(dec_c2, dec_c1)
		self.dec0 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
			ConvBNReLU(dec_c1, dec_c1),
			scSE(dec_c1),
		)
		# Separate heads
		self.retina_head = nn.Conv2d(dec_c1, 1, 1)
		self.nv_head = nn.Conv2d(dec_c1, 1, 1)
		self.vo_head = nn.Conv2d(dec_c1, 1, 1)

		# Deep supervision heads at y1 (dec1 output) and y2 (dec2 output)
		self.retina_ds2_head = nn.Conv2d(dec_c2, 1, 1)
		self.nv_ds2_head = nn.Conv2d(dec_c2, 1, 1)
		self.vo_ds2_head = nn.Conv2d(dec_c2, 1, 1)
		self.retina_ds3_head = nn.Conv2d(dec_c3, 1, 1)
		self.nv_ds3_head = nn.Conv2d(dec_c3, 1, 1)
		self.vo_ds3_head = nn.Conv2d(dec_c3, 1, 1)

	def freeze_retina_head(self):
		for p in self.retina_head.parameters():
			p.requires_grad = False

	def freeze_retina_heads(self):
		for name, p in self.named_parameters():
			if ("retina_head" in name) or ("retina_ds" in name):
				p.requires_grad = False

	def reset_nv_vo_heads(self):
		for m in [self.nv_head, self.vo_head]:
			nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		for m in [self.nv_ds2_head, self.vo_ds2_head, self.nv_ds3_head, self.vo_ds3_head]:
			nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
			if m.bias is not None:
				nn.init.zeros_(m.bias)

	def _decoder_block(self, in_c: int, skip_c: int):
		return nn.ModuleDict({
			"up": nn.Sequential(
				nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
				nn.Conv2d(in_c, skip_c, 1, bias=False),
			),
			"att": AttentionGate(skip_c, skip_c, inter_ch=max(skip_c // 2, 1)),
			"conv": nn.Sequential(
				ConvBNReLU(skip_c * 2, skip_c),
				ConvBNReLU(skip_c, skip_c),
				scSE(skip_c),
			),
		})

	def forward(self, x):
		# Encoder features
		f1, f2, f3, f4 = self.enc(x)
		p1 = self.proj1(f1)
		p2 = self.proj2(f2)
		p3 = self.proj3(f3)
		p4 = self.proj4(f4)

		y3_u = self.dec3["up"](p4)
		y3_s = self.dec3["att"](p3, y3_u)
		y3 = self.dec3["conv"](torch.cat([y3_u, y3_s], dim=1).contiguous())

		y2_u = self.dec2["up"](y3)
		y2_s = self.dec2["att"](p2, y2_u)
		y2 = self.dec2["conv"](torch.cat([y2_u, y2_s], dim=1).contiguous())

		y1_u = self.dec1["up"](y2)
		y1_s = self.dec1["att"](p1, y1_u)
		y1 = self.dec1["conv"](torch.cat([y1_u, y1_s], dim=1).contiguous())

		y0 = self.dec0(y1.contiguous())
		# y0 is typically 256x256 for 512 input; upsample heads to 512
		logits_r = F.interpolate(self.retina_head(y0), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		logits_nv = F.interpolate(self.nv_head(y0), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		logits_vo = F.interpolate(self.vo_head(y0), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		logits = torch.cat([logits_r, logits_nv, logits_vo], dim=1)
		return logits

	def forward_with_aux(self, x):
		# Encoder features
		f1, f2, f3, f4 = self.enc(x)
		p1 = self.proj1(f1)
		p2 = self.proj2(f2)
		p3 = self.proj3(f3)
		p4 = self.proj4(f4)

		y3_u = self.dec3["up"](p4)
		y3_s = self.dec3["att"](p3, y3_u)
		y3 = self.dec3["conv"](torch.cat([y3_u, y3_s], dim=1).contiguous())

		y2_u = self.dec2["up"](y3)
		y2_s = self.dec2["att"](p2, y2_u)
		y2 = self.dec2["conv"](torch.cat([y2_u, y2_s], dim=1).contiguous())

		y1_u = self.dec1["up"](y2)
		y1_s = self.dec1["att"](p1, y1_u)
		y1 = self.dec1["conv"](torch.cat([y1_u, y1_s], dim=1).contiguous())

		y0 = self.dec0(y1.contiguous())
		logits_r = F.interpolate(self.retina_head(y0), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		logits_nv = F.interpolate(self.nv_head(y0), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		logits_vo = F.interpolate(self.vo_head(y0), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		main = torch.cat([logits_r, logits_nv, logits_vo], dim=1)

		# ds2 from y2
		ds2_r = F.interpolate(self.retina_ds2_head(y2), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds2_nv = F.interpolate(self.nv_ds2_head(y2), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds2_vo = F.interpolate(self.vo_ds2_head(y2), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds2 = torch.cat([ds2_r, ds2_nv, ds2_vo], dim=1)
		# ds3 from y3
		ds3_r = F.interpolate(self.retina_ds3_head(y3), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds3_nv = F.interpolate(self.nv_ds3_head(y3), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds3_vo = F.interpolate(self.vo_ds3_head(y3), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds3 = torch.cat([ds3_r, ds3_nv, ds3_vo], dim=1)
		return {"main": main, "ds2": ds2, "ds3": ds3}
