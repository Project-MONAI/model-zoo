from torchvision import models
import torch.nn as nn
import torch


class ModelDefinition():
	def __init__(
		self,
		num_class =4,
		pretrained_flag= 0,
		dropout_ratio= 0.5,
		fc_nodes=int):

		super().__init__()
		model = models.inception_v2(pretrained = pretrained_flag)
		model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_class))
		model.aux_logits = False

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.class_layers(x)
		return x
