import torch
import torch.nn as nn
import timm

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(VisionTransformer, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x