import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNet(nn.Module):
    """
    ResNet50 backbone
    """

    def __init__(self, num_classes=100, embedding_dim=1024, small_input=True, pretrained: bool = True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_model = models.resnet50(weights=weights)

        # CIFAR 32×32 small input
        if small_input:
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base_model.maxpool = nn.Identity()

        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # [B, 2048, 1, 1]

        self.emb_dim_fix_layer = nn.Sequential(
            nn.Linear(base_model.fc.in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        feat_map = self.backbone(x)  # [B, 2048, 1, 1]
        feat = torch.flatten(feat_map, 1)  # [B, 2048]
        embedding = self.emb_dim_fix_layer(feat)  # [B, 1024]
        # embedding = F.normalize(embedding, dim=1)
        out = self.classifier(embedding)
        return embedding, out
