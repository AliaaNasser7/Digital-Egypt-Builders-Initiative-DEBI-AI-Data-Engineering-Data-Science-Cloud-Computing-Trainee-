import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

@st.cache_resource
def load_resources():
    model = load_model("plant_disease_model.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_resources()

# UI
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a leaf image and the model will detect if it\'s healthy or diseased.")
st.divider()

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    img_resized = img.resize((128, 128))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    with st.spinner("Analyzing..."):
        preds = model.predict(arr)
        pred_idx   = np.argmax(preds)
        pred_label = class_names[pred_idx]
        confidence = preds[0][pred_idx] * 100

    st.divider()

    if "healthy" in pred_label.lower():
        st.success(f"**{pred_label.replace('_', ' ')}**")
    else:
        st.error(f"**{pred_label.replace('_', ' ')}**")

    st.metric("Confidence", f"{confidence:.2f}%")

    st.divider()
    st.subheader("Top 5 Predictions")
    top5_idx  = np.argsort(preds[0])[::-1][:5]
    for i, idx in enumerate(top5_idx):
        label = class_names[idx].replace('_', ' ')
        prob  = preds[0][idx] * 100
        st.progress(int(prob), text=f"{i+1}. {label} — {prob:.2f}%")

#streamlit run app.py'