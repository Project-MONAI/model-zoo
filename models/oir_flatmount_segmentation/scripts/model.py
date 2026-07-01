from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
	def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
		super().__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
		self.bn = nn.BatchNorm2d(out_ch)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = x.contiguous()
		x = self.conv(x).contiguous()
		x = self.bn(x)
		x = self.relu(x)
		return x


class scSE(nn.Module):
	# Concurrent spatial and channel squeeze & excitation
	def __init__(self, ch: int, reduction: int = 16):
		super().__init__()
		self.cSE = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(ch, max(ch // reduction, 1), 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(max(ch // reduction, 1), ch, 1),
			nn.Sigmoid(),
		)
		self.sSE = nn.Sequential(
			nn.Conv2d(ch, 1, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		c = self.cSE(x)
		s = self.sSE(x)
		return x * c + x * s


class AttentionGate(nn.Module):
	# Gating from decoder g modulates skip x
	def __init__(self, in_ch_x: int, in_ch_g: int, inter_ch: int):
		super().__init__()
		self.theta_x = nn.Conv2d(in_ch_x, inter_ch, kernel_size=1, bias=False)
		self.phi_g = nn.Conv2d(in_ch_g, inter_ch, kernel_size=1, bias=True)
		self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x_skip, g):
		# Both x_skip and g should have the same spatial size here
		theta_x = self.theta_x(x_skip)
		phi_g = self.phi_g(g)
		t = self.relu(theta_x + phi_g)
		psi = self.sigmoid(self.psi(t))
		return x_skip * psi


class EncoderBlock(nn.Module):
	def __init__(self, in_ch: int, out_ch: int):
		super().__init__()
		self.conv1 = ConvBNReLU(in_ch, out_ch)
		self.conv2 = ConvBNReLU(out_ch, out_ch)
		self.scse = scSE(out_ch)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.scse(x)
		return x


class DecoderBlock(nn.Module):
	def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
		super().__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
			nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
		)
		# Gate uses decoder's out_ch as g input channels
		self.att = AttentionGate(skip_ch, out_ch, inter_ch=max(out_ch // 2, 1))
		self.conv1 = ConvBNReLU(out_ch + skip_ch, out_ch)
		self.conv2 = ConvBNReLU(out_ch, out_ch)
		self.scse = scSE(out_ch)

	def forward(self, x, skip):
		x = self.up(x)
		skip = self.att(skip, x)
		x = torch.cat([x, skip], dim=1).contiguous()
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.scse(x)
		return x


class AttentionUNet(nn.Module):
	def __init__(self, in_ch: int = 1, base_ch: int = 32, out_ch: int = 3):
		super().__init__()
		c1 = base_ch
		c2 = base_ch * 2
		c3 = base_ch * 4
		c4 = base_ch * 8
		c5 = base_ch * 16

		self.enc1 = EncoderBlock(in_ch, c1)
		self.enc2 = EncoderBlock(c1, c2)
		self.enc3 = EncoderBlock(c2, c3)
		self.enc4 = EncoderBlock(c3, c4)
		self.center = EncoderBlock(c4, c5)

		self.pool = nn.MaxPool2d(2)

		self.dec4 = DecoderBlock(c5, c4, c4)
		self.dec3 = DecoderBlock(c4, c3, c3)
		self.dec2 = DecoderBlock(c3, c2, c2)
		self.dec1 = DecoderBlock(c2, c1, c1)

		# Separate heads for retina, NV, VO (main output at d1)
		self.retina_head = nn.Conv2d(c1, 1, kernel_size=1)
		self.nv_head = nn.Conv2d(c1, 1, kernel_size=1)
		self.vo_head = nn.Conv2d(c1, 1, kernel_size=1)

		# Deep supervision heads at d2 and d3
		self.retina_ds2_head = nn.Conv2d(c2, 1, kernel_size=1)
		self.nv_ds2_head = nn.Conv2d(c2, 1, kernel_size=1)
		self.vo_ds2_head = nn.Conv2d(c2, 1, kernel_size=1)
		self.retina_ds3_head = nn.Conv2d(c3, 1, kernel_size=1)
		self.nv_ds3_head = nn.Conv2d(c3, 1, kernel_size=1)
		self.vo_ds3_head = nn.Conv2d(c3, 1, kernel_size=1)

	def freeze_retina_head(self):
		# Backward compatibility
		for p in self.retina_head.parameters():
			p.requires_grad = False

	def freeze_retina_heads(self):
		# Freeze all retina-related heads including deep supervision
		for name, p in self.named_parameters():
			if ("retina_head" in name) or ("retina_ds" in name):
				p.requires_grad = False

	def reset_nv_vo_heads(self):
		# Reset main heads
		for m in [self.nv_head, self.vo_head]:
			nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		# Reset DS heads
		for m in [self.nv_ds2_head, self.vo_ds2_head, self.nv_ds3_head, self.vo_ds3_head]:
			nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
			if m.bias is not None:
				nn.init.zeros_(m.bias)

	def forward(self, x):
		e1 = self.enc1(x)
		e2 = self.enc2(self.pool(e1))
		e3 = self.enc3(self.pool(e2))
		e4 = self.enc4(self.pool(e3))
		cent = self.center(self.pool(e4))

		d4 = self.dec4(cent, e4)
		d3 = self.dec3(d4, e3)
		d2 = self.dec2(d3, e2)
		d1 = self.dec1(d2, e1)

		logits_r = self.retina_head(d1)
		logits_nv = self.nv_head(d1)
		logits_vo = self.vo_head(d1)
		logits = torch.cat([logits_r, logits_nv, logits_vo], dim=1)
		return logits

	def forward_with_aux(self, x):
		# Returns main logits and deep supervision logits upsampled to input size
		e1 = self.enc1(x)
		e2 = self.enc2(self.pool(e1))
		e3 = self.enc3(self.pool(e2))
		e4 = self.enc4(self.pool(e3))
		cent = self.center(self.pool(e4))

		d4 = self.dec4(cent, e4)
		d3 = self.dec3(d4, e3)
		d2 = self.dec2(d3, e2)
		d1 = self.dec1(d2, e1)

		# main
		logits_r = self.retina_head(d1)
		logits_nv = self.nv_head(d1)
		logits_vo = self.vo_head(d1)
		main = torch.cat([logits_r, logits_nv, logits_vo], dim=1)
		# ds2 from d2
		ds2_r = F.interpolate(self.retina_ds2_head(d2), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds2_nv = F.interpolate(self.nv_ds2_head(d2), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds2_vo = F.interpolate(self.vo_ds2_head(d2), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds2 = torch.cat([ds2_r, ds2_nv, ds2_vo], dim=1)
		# ds3 from d3
		ds3_r = F.interpolate(self.retina_ds3_head(d3), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds3_nv = F.interpolate(self.nv_ds3_head(d3), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds3_vo = F.interpolate(self.vo_ds3_head(d3), size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		ds3 = torch.cat([ds3_r, ds3_nv, ds3_vo], dim=1)
		return {"main": main, "ds2": ds2, "ds3": ds3}
