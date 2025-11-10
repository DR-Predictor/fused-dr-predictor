# 🧠 Fusion DR Predictor (MVP)

## 📖 Project Overview
**Fusion DR Predictor** is a machine learning-based diagnostic tool designed to **predict the severity of Diabetic Retinopathy (DR)** by fusing retinal image features with patient clinical data.  
This MVP (Minimum Viable Product) supports **early detection**, **risk assessment**, and **personalized follow-up recommendations**, empowering healthcare providers to make informed decisions.

---

## 🏗️ Model Development Workflow

### 1. Problem Definition
The model classifies Diabetic Retinopathy (DR) severity into five categories:
- **No DR**
- **Mild NPDR**
- **Moderate NPDR**
- **Severe NPDR**
- **Proliferative DR (PDR)**

Additionally, the system provides **follow-up recommendations** based on predicted risk levels.

---

### 2. Data Sources
- **Image Data:** Retinal fundus images (preprocessed and resized)  
- **Clinical Data:** Three key patient features:
  - HbA1c level  
  - Blood pressure  
  - Duration of diabetes  

---

### 3. Image Feature Extraction
A **pretrained ResNet50** model from `torchvision` was utilized to extract deep image features:
- The final classification layer (`resnet.fc`) was replaced with `Identity()`
- The resulting feature vector per image is **2048-dimensional**

This approach leverages **transfer learning**, allowing the model to capture rich visual representations without retraining the full CNN.

---

### 4. Fusion Model Architecture
The model fuses **image** and **clinical** features into a single prediction pipeline:

```
[Image Features: 2048-d] + [Clinical Features: 3-d]
              ↓
       Concatenated Vector
              ↓
   Fully Connected Layers
              ↓
        Softmax Output (5 Classes)
```

The model is trained using:
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam  

---

### 5. Risk Mapping Logic
A custom module, `risk_mapper.py`, maps predicted classes to actionable follow-up recommendations:

| DR Grade         | Risk Level | Follow-Up Recommendation   |
|------------------|-------------|-----------------------------|
| No DR            | Low         | 12 Months                  |
| Mild NPDR        | Low         | 12 Months                  |
| Moderate NPDR    | Moderate    | 6 Months                   |
| Severe NPDR      | High        | 3 Months                   |
| PDR              | Critical    | Immediate Referral          |

This ensures **clinical interpretability** and **real-world utility** of predictions.

---

### 6. Streamlit MVP Interface
A simple **Streamlit** application (`app.py`) enables interactive testing:
- Upload a retinal image  
- Input clinical parameters  
- Run prediction  
- View:
  - DR Grade  
  - Probability Vector  
  - Follow-Up Recommendation  

This interface is ideal for **demo**, **validation**, and **stakeholder feedback**.

---

## 📦 Project Structure

```
fusion-dr-predictor/
├── app.py                  # Streamlit interface
├── fusion_model.py         # Fusion model definition
├── utils.py                # ResNet feature extraction utilities
├── risk_mapper.py          # Risk mapping logic
├── fusion_model_mvp.pth    # Trained model weights
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 👥 Contributors
- **Almustapha Damilola Usman** — *Machine Learning Engineer*  
  *(Model design, training, and integration)*  
- **Team Members** — *Web Development, UI/UX, and Deployment*  
  *(To be added post-launch)*

---

## 🚀 Future Improvements
- Integration with cloud-based medical imaging APIs  
- Deployment to scalable web infrastructure  
- Incorporation of explainable AI (Grad-CAM visualizations)  
- Model fine-tuning with larger, diverse datasets  

---

## 🧩 Tech Stack
- **Language:** Python  
- **Libraries:** PyTorch, Torchvision, Streamlit, NumPy, Pandas  
- **Model:** ResNet50 (Transfer Learning)  
- **Interface:** Streamlit  
- **Deployment Target:** MVP (Web Demo)

---

> *Fusion DR Predictor aims to bridge the gap between medical imaging and intelligent decision support — ensuring timely and accurate DR diagnosis for better patient outcomes.*
