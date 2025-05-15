import streamlit as st
from PIL import Image
from utils import load_model, predict_disease

# Load model and class names
model, class_names = load_model()

# Set Streamlit page configuration
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection System for Sustainable Agriculture")
page = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# ---- PAGE: HOME ----
if page == "HOME":
    st.image("your_image.jpg", use_column_width=True)
    st.markdown("""
        ## Plant Disease Detection System for Sustainable Agriculture
        Welcome! This tool allows you to detect plant diseases using AI.  
        Select 'Disease Recognition' from the menu to get started.
    """)

# ---- PAGE: DISEASE RECOGNITION ----
elif page == "DISEASE RECOGNITION":
    st.title("Sustainable Agriculture")
    st.markdown("### Choose an Image:")

    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.success(f"File uploaded: {uploaded_file.name}")
        image = Image.open(uploaded_file).convert('RGB')

        if st.button("Show Image"):
            st.image(image, caption="Uploaded Image", width=300)

        if st.button("Predict"):
            label, confidence = predict_disease(image, model, class_names)
            st.markdown("## Our Prediction")
            st.success(f"**{label}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
