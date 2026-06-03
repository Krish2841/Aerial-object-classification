import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Page config
st.set_page_config(
    page_title="Aerial Object Classifier",
    page_icon="🛰️",
    layout="centered"
)

# Load model
model = load_model("../models/final_model.h5")

# Sidebar
st.sidebar.title("About Project")
st.sidebar.info(
    """
    This application classifies aerial images into:
    - 🐦 Bird
    - 🛸 Drone

    Model Used:
    MobileNetV2 Transfer Learning
    """
)

# Main title
st.markdown(
    "<h1 style='text-align:center;'>🛰️ Aerial Object Classification</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Upload an aerial image and detect whether it is a Bird or Drone.</p>",
    unsafe_allow_html=True
)

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    resized = img.resize((224, 224))
    img_array = np.array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "🛸 Drone"
        confidence = prediction * 100
    else:
        label = "🐦 Bird"
        confidence = (1 - prediction) * 100

    with col2:
        st.subheader("Prediction")
        st.success(label)
        st.metric("Confidence", f"{confidence:.2f}%")

# Footer
st.markdown("---")
st.caption("Developed using TensorFlow + Streamlit")
