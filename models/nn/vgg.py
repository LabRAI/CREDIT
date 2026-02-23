import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_BN_Weights


class VGG(nn.Module):
    """
    VGG16 backbone
    """

    def __init__(self, num_classes=100, embedding_dim=1024, small_input=True, pretrained=True):
        super().__init__()

        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        base = models.vgg16_bn(weights=weights)

        base.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if small_input:
            feats = list(base.features)
            pool_idxs = [i for i, m in enumerate(feats) if isinstance(m, nn.MaxPool2d)]
            if pool_idxs:
                feats[pool_idxs[0]] = nn.Identity()
            base.features = nn.Sequential(*feats)

        self.backbone = nn.Sequential(base.features, base.avgpool)

        self.emb_dim_fix_layer = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        feat_map = self.backbone(x)  # [B, 512, 1, 1]
        feat = torch.flatten(feat_map, 1)  # [B, 512]
        embedding = self.emb_dim_fix_layer(feat)  # [B, 1024]
        # embedding = F.normalize(embedding, dim=1)
        out = self.classifier(embedding)
        return embedding, out
