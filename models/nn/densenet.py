import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights


class DenseNet(nn.Module):
    """
    DenseNet121 backbone
    """

    def __init__(self, num_classes=100, embedding_dim=1024, small_input=True, pretrained=True):
        super().__init__()

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base = models.densenet121(weights=weights)

        if small_input:
            base.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.features.pool0 = nn.Identity()

        self.features = base.features
        self.norm_relu = nn.Sequential(nn.BatchNorm2d(1024), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding_dim = base.classifier.in_features  # 1024
        assert self.embedding_dim == embedding_dim

        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        feat = self.features(x)  # [B, 1024, H, W]
        feat = self.norm_relu(feat)
        feat = self.pool(feat)  # [B, 1024, 1, 1]
        embedding = torch.flatten(feat, 1)  # [B, 1024]
        # embedding = F.normalize(embedding, dim=1)
        out = self.classifier(embedding)
        return embedding, out
