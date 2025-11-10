# app.py

import streamlit as st
import torch
from PIL import Image
import numpy as np
from fusion_model import FusionModel
from utils import extract_image_features
from risk_mapper import map_risk

# === Load Model ===
device = torch.device("cpu")
model = FusionModel()
model.load_state_dict(torch.load("fusion_model_mvp.pth", map_location=device))
model.to(device)
model.eval()

# === UI ===
st.title("Diabetic Retinopathy Risk Predictor")

uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "jpeg", "png"])
hba1c = st.number_input("HbA1c", min_value=0.0, max_value=15.0, step=0.1)
duration = st.number_input("Duration of Diabetes (years)", min_value=0.0, max_value=50.0, step=0.5)
bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, step=1)

if uploaded_file and st.button("Predict Risk"):
    image = Image.open(uploaded_file).convert("RGB")
    image_tensor = extract_image_features(image).to(device)
    clinical_tensor = torch.tensor([[hba1c, bp, duration]], dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(image_tensor, clinical_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_index = int(np.argmax(probabilities))
        recommendation = map_risk(pred_index)

    st.subheader(f"Prediction: {['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR'][pred_index]}")
    st.write(f"Probability Vector: {np.round(probabilities, 3)}")
    st.success(recommendation)