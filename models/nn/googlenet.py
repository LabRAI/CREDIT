import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import GoogLeNet_Weights


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) backbone
    """

    def __init__(self, num_classes=100, embedding_dim=1024, small_input=True, pretrained=True):
        super().__init__()

        weights = GoogLeNet_Weights.DEFAULT if pretrained else None
        base = models.googlenet(weights=weights, aux_logits=True)

        if small_input:
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool1 = nn.Identity()

        # backbone: rm fc layer
        self.backbone = nn.Sequential(
            base.conv1, base.maxpool1,
            base.conv2, base.conv3, base.maxpool2,
            base.inception3a, base.inception3b, base.maxpool3,
            base.inception4a, base.inception4b, base.inception4c,
            base.inception4d, base.inception4e, base.maxpool4,
            base.inception5a, base.inception5b,
            base.avgpool, nn.Flatten()
        )

        self.embedding_dim = base.fc.in_features  # 1024
        assert self.embedding_dim == embedding_dim

        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        embedding = self.backbone(x)  # [B, 1024]
        # embedding = F.normalize(embedding, dim=1)
        out = self.classifier(embedding)  # [B, num_classes]
        return embedding, out
