
import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig

class VisionEncoder(nn.Module):
    def __init__(self, image_size=224, backbone='vit_base_patch16_224', pretrained=True, cnn_stem=True, out_dim=768):
        super().__init__()
        self.cnn_stem = None
        if cnn_stem:
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            in_ch = 32
        else:
            in_ch = 4
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=in_ch, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x):
        if self.cnn_stem is not None:
            x = self.cnn_stem(x)
        feat = self.backbone(x)
        return self.proj(feat)

class TextEncoder(nn.Module):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', out_dim=768, freeze_layers=0):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=cfg)
        # Optionally freeze some lower layers
        if freeze_layers > 0:
            n_freeze = freeze_layers
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < n_freeze:
                    for p in layer.parameters():
                        p.requires_grad = False
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:,0]  # CLS
        return self.proj(pooled)
