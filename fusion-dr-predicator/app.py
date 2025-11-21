import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define the model architecture (only fusion parts, NOT feature extractor)
class FusionModel(nn.Module):
    def __init__(self, num_classes=5, clinical_features=3):
        super(FusionModel, self).__init__()
        
        # Image branch - Simple linear layer on extracted features
        # Input: 2048 (pre-extracted ResNet features) -> Output: 256
        self.image_branch = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU()
        )
        
        # Clinical data branch
        # Input: 3 clinical features -> Output: 32
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_features, 32),
            nn.ReLU()
        )
        
        # Fusion layers (256 from image + 32 from clinical = 288 total)
        self.fusion = nn.Sequential(
            nn.Linear(288, 128),  # 256 + 32 = 288 inputs
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output 5 classes
        )
    
    def forward(self, img_features, clinical):
        # Process through image branch
        img_out = self.image_branch(img_features)
        
        # Process clinical data
        clinical_out = self.clinical_branch(clinical)
        
        # Concatenate and fuse
        combined = torch.cat((img_out, clinical_out), dim=1)  # [batch, 288]
        output = self.fusion(combined)
        
        return output

# Load the fusion model
fusion_model = FusionModel(num_classes=5, clinical_features=3)
fusion_model.load_state_dict(torch.load('fusion_model_mvp.pth', map_location=torch.device('cpu')))
fusion_model.eval()

# Load ResNet50 feature extractor separately (with pretrained ImageNet weights)
resnet = models.resnet50(weights='IMAGENET1K_V1')
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

# Freeze feature extractor
for param in feature_extractor.parameters():
    param.requires_grad = False

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# DR class labels and recommendations
DR_CLASSES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

RECOMMENDATIONS = {
    0: "Annual screening recommended. Maintain good glycemic control.",
    1: "Follow-up in 9-12 months. Monitor blood sugar and blood pressure closely.",
    2: "Follow-up in 6-9 months. Consider referral to ophthalmologist.",
    3: "Follow-up in 3-4 months. Urgent ophthalmologist referral recommended.",
    4: "Immediate ophthalmologist referral required. High risk of vision loss."
}

# Prediction function
def predict_dr(image, hba1c, blood_pressure, duration):
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        else:
            image = image.convert('RGB')
        
        # Preprocess image
        img_tensor = preprocess_image(image)
        
        # Extract features using ResNet50
        with torch.no_grad():
            img_features = feature_extractor(img_tensor)
            img_features = img_features.view(img_features.size(0), -1)  # Flatten to [batch, 2048]
        
        # Prepare clinical data
        clinical_data = torch.tensor([[hba1c, blood_pressure, duration]], dtype=torch.float32)
        
        # Make prediction with fusion model
        with torch.no_grad():
            output = fusion_model(img_features, clinical_data)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        # Format results
        severity = DR_CLASSES[predicted_class]
        recommendation = RECOMMENDATIONS[predicted_class]
        
        # Create probability dictionary for all classes
        prob_dict = {DR_CLASSES[i]: float(probabilities[0][i].item()) for i in range(5)}
        
        # Result text
        result_text = f"""
## 🔍 Diagnosis Results

**Predicted Severity:** {severity}  
**Confidence:** {confidence:.1f}%

---

### 📋 Recommendation:
{recommendation}

---

### 📊 Clinical Input Summary:
- **HbA1c Level:** {hba1c}%
- **Blood Pressure:** {blood_pressure} mmHg
- **Diabetes Duration:** {duration} years

---

⚠️ **Disclaimer:** This is an AI-assisted diagnostic tool for research and educational purposes. It should not replace professional medical judgment. Always consult with a qualified ophthalmologist for clinical decisions.
        """
        
        return result_text, prob_dict
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}\n\nPlease ensure the image is a valid retinal fundus photograph.", {}

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="RetinaCare DR Classifier") as demo:
    gr.Markdown(
        """
        # 👁️ RetinaCare: DR Severity Classifier
        **AI-powered Diabetic Retinopathy Detection using Multimodal Fusion**
        
        Upload a retinal fundus image and provide clinical information to get an AI-powered DR severity assessment.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Retinal Image")
            image_input = gr.Image(
                type="pil",
                label="Upload Retinal Fundus Image",
                height=300
            )
            
            gr.Markdown("### 📋 Clinical Information")
            hba1c_input = gr.Slider(
                minimum=4.0,
                maximum=15.0,
                value=7.0,
                step=0.1,
                label="HbA1c Level (%)",
                info="Glycated hemoglobin level (normal: 4-5.6%)"
            )
            
            bp_input = gr.Slider(
                minimum=80,
                maximum=200,
                value=120,
                step=1,
                label="Systolic Blood Pressure (mmHg)",
                info="Systolic blood pressure reading"
            )
            
            duration_input = gr.Slider(
                minimum=0,
                maximum=50,
                value=5,
                step=1,
                label="Duration of Diabetes (years)",
                info="Years since diabetes diagnosis"
            )
            
            predict_btn = gr.Button("🔍 Analyze Retinal Image", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Results")
            result_output = gr.Markdown(label="Diagnosis")
            
            gr.Markdown("### 📈 Probability Distribution")
            prob_output = gr.Label(label="Class Probabilities", num_top_classes=5)
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_dr,
        inputs=[image_input, hba1c_input, bp_input, duration_input],
        outputs=[result_output, prob_output]
    )
    
    gr.Markdown(
        """
        ---
        ### 🏥 DR Severity Classification Guide:
        - **No DR:** No visible signs of diabetic retinopathy
        - **Mild NPDR:** Presence of microaneurysms only
        - **Moderate NPDR:** More than just microaneurysms, but less than severe NPDR
        - **Severe NPDR:** Extensive hemorrhages, venous beading, or IRMA
        - **Proliferative DR:** Growth of new blood vessels (highest risk of vision loss)
        
        ---
        *RetinaCare DR Classifier | Powered by Deep Learning | For research and educational purposes*
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
