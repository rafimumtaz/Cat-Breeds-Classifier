import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import time
import os

st.set_page_config(page_title="Cat Breeds Classifier", page_icon="🐱", layout="centered")

# Inject custom CSS for premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1f1c2c 0%, #928DAB 100%);
        color: white;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #FFE66D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #E0E0E0;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 43, 0.6);
        color: white;
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .breed-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFE66D;
        margin: 10px 0;
        text-transform: capitalize;
    }
    
    .confidence {
        font-size: 1.2rem;
        color: #A8E6CF;
    }
    
    /* Improve file uploader look */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255,255,255,0.05);
        border: 2px dashed rgba(255,255,255,0.3);
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">FelineVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Discover the breed of any cat instantly using Vision Transformers</div>', unsafe_allow_html=True)

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        return None, None, None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(model_path, 'rb') as f:
        save_data = pickle.load(f)
        
    class_names = save_data['class_names']
    num_classes = len(class_names)
    
    # Reconstruct the model architecture
    model = models.vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # Load weights
    model.load_state_dict(save_data['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, device

model_path = "cat_breeds_vit.pkl"
model, class_names, device = load_model(model_path)

if model is None:
    st.error("Model file not found! Please run `train_model.py` first to generate `cat_breeds_vit.pkl`.")
    st.info("The application will become fully functional once the model is trained and saved.")
    st.stop()

uploaded_file = st.file_uploader("Upload a photo of a cat...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
    if st.button("Identify Breed 🐾", use_container_width=True):
        with st.spinner("Analyzing feline features..."):
            # Simulate a small delay for dramatic effect
            time.sleep(1)
            
            # Preprocess
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            predicted_breed = class_names[predicted_idx.item()].replace('_', ' ')
            confidence_pct = confidence.item() * 100
            
            # Display Result
            st.markdown(f"""
            <div class="result-card">
                <p style="font-size: 1.1rem; color: #FFF; margin: 0;">Our AI thinks this is a</p>
                <div class="breed-title">{predicted_breed}</div>
                <div class="confidence">Confidence: {confidence_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
