import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# === Model Definition ===
class FusionModel(nn.Module):
    def __init__(self, image_feature_size=2048, clinical_feature_size=3, num_classes=5):
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

# === Load Model ===
device = torch.device("cpu")
model = FusionModel()
model.load_state_dict(torch.load("fusion_model_mvp.pth", map_location=device))
model.to(device)
model.eval()

# === Example Inputs ===
# Replace this with actual image features (e.g., from a CNN)
image_tensor = torch.randn(1, 2048).to(device)

# Replace with real clinical data: [HbA1c, BP, Duration]
clinical_tensor = torch.tensor([[7.2, 130, 5]], dtype=torch.float32).to(device)

# === Predict ===
with torch.no_grad():
    output = model(image_tensor, clinical_tensor)
    _, predicted = torch.max(output, 1)

print(f"Predicted DR grade: {predicted.item()}")