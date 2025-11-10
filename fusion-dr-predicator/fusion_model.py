# fusion_model.py

import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(
        self,
        image_feature_size=2048,
        clinical_feature_size=3,
        num_classes=5,
    ):
        super(FusionModel, self).__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(image_feature_size, 256),
            nn.ReLU()
        )
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_feature_size, 32),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image_features, clinical_features):
        image_out = self.image_branch(image_features)
        clinical_out = self.clinical_branch(clinical_features)
        combined = torch.cat((image_out, clinical_out), dim=1)
        return self.fusion(combined)