import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define the model architecture (same as training)
class FusionModel(nn.Module):
    def __init__(self, num_classes=5, clinical_features=3):
        super(FusionModel, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=False)
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Fusion layers
        self.fc1 = nn.Linear(2048 + clinical_features, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, image, clinical):
        # Extract image features
        img_features = self.resnet(image)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Concatenate image features with clinical data
        combined = torch.cat((img_features, clinical), dim=1)
        
        # Pass through fusion layers
        x = self.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# Load model
@st.cache_resource
def load_model():
    model = FusionModel(num_classes=5, clinical_features=3)
    model.load_state_dict(torch.load('fusion_model_mvp.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

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

# Streamlit UI
st.set_page_config(page_title="RetinaCare DR Classifier", page_icon="👁️", layout="centered")

st.title("👁️ RetinaCare: DR Severity Classifier")
st.markdown("**Multimodal AI-powered Diabetic Retinopathy Detection**")
st.markdown("---")

# Sidebar for clinical inputs
st.sidebar.header("📋 Clinical Information")
hba1c = st.sidebar.number_input(
    "HbA1c Level (%)", 
    min_value=4.0, 
    max_value=15.0, 
    value=7.0, 
    step=0.1,
    help="Glycated hemoglobin level (normal: 4-5.6%)"
)

blood_pressure = st.sidebar.number_input(
    "Systolic Blood Pressure (mmHg)", 
    min_value=80, 
    max_value=200, 
    value=120, 
    step=1,
    help="Systolic blood pressure reading"
)

duration = st.sidebar.number_input(
    "Duration of Diabetes (years)", 
    min_value=0, 
    max_value=50, 
    value=5, 
    step=1,
    help="Years since diabetes diagnosis"
)

# Main content area
st.header("📸 Upload Retinal Image")
uploaded_file = st.file_uploader(
    "Choose a retinal fundus image", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a color fundus photograph of the retina"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Clinical Data")
        st.metric("HbA1c", f"{hba1c}%")
        st.metric("Blood Pressure", f"{blood_pressure} mmHg")
        st.metric("Diabetes Duration", f"{duration} years")
    
    # Predict button
    if st.button("🔍 Analyze Retinal Image", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # Load model
                model = load_model()
                
                # Preprocess image
                img_tensor = preprocess_image(image)
                
                # Prepare clinical data
                clinical_data = torch.tensor([[hba1c, blood_pressure, duration]], dtype=torch.float32)
                
                # Make prediction
                with torch.no_grad():
                    output = model(img_tensor, clinical_data)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item() * 100
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Prediction with color coding
                severity = DR_CLASSES[predicted_class]
                colors = {
                    "No DR": "green",
                    "Mild NPDR": "blue",
                    "Moderate NPDR": "orange",
                    "Severe NPDR": "red",
                    "Proliferative DR": "darkred"
                }
                
                st.markdown(f"### Predicted Severity: :{colors[severity]}[{severity}]")
                st.metric("Confidence", f"{confidence:.1f}%")
                
                # Recommendation
                st.info(f"**Recommendation:** {RECOMMENDATIONS[predicted_class]}")
                
                # Show probability distribution
                st.subheader("Probability Distribution")
                prob_dict = {DR_CLASSES[i]: f"{probabilities[0][i].item()*100:.1f}%" for i in range(5)}
                st.bar_chart({k: float(v.rstrip('%')) for k, v in prob_dict.items()})
                
                # Disclaimer
                st.warning("⚠️ **Disclaimer:** This is an AI-assisted diagnostic tool and should not replace professional medical judgment. Always consult with a qualified ophthalmologist for clinical decisions.")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please ensure the image is a valid retinal fundus photograph.")

else:
    st.info("👆 Please upload a retinal image to begin analysis")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: gray; font-size: 0.9em'>
            RetinaCare DR Classifier | Powered by Deep Learning<br>
            For research and educational purposes
        </p>
    </div>
    """,
    unsafe_allow_html=True
)